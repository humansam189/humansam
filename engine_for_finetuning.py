import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt


def train_class_batch(model, samples, depths, target, criterion, confs):
    outputs = model(samples, depths)
    loss = criterion(outputs, target)
    weights = confs

    # Multiply the loss by the corresponding weights
    weighted_losses = loss * weights

    # Calculate the average of the weighted losses
    loss = torch.mean(weighted_losses)
    return loss, outputs


# def train_class_batch(model, samples, target, criterion):
#     outputs = model(samples)
#     loss = criterion(outputs, target)
#     return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
        start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None,
        bf16=False,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, confs, depths) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # confs = torch.cat(confs, dim=0)
        confs = torch.log(confs + torch.exp(torch.ones(1)))
        confs = confs.to(device, non_blocking=True)
        depths = torch.cat(depths, dim=0)
        depths = depths.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.bfloat16() if bf16 else samples.half()
            depths = depths.half()
            loss, output = train_class_batch(
                model, samples, depths, targets, criterion)
            # loss, output = train_class_batch(
            #     model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                depths = depths.half()
                loss, output = train_class_batch(
                    model, samples, depths, targets, criterion, confs)
                # loss, output = train_class_batch(
                #     model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, ds=False, bf16=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        depth = batch[2]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        depths = depth.to(device, non_blocking=True)

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            depths = depths.half()
            output = model(videos, depths)
            # output = model(videos)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos, depths)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, ds=False, bf16=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        depth = batch[5]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        depths = depth.to(device, non_blocking=True)

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            depths = depths.half()
            output = model(videos, depths)
            # output = model(videos)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos, depths)
                # output = model(videos)
                loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                               str(output.data[i].float().cpu().numpy().tolist()), \
                                               str(int(target[i].cpu().numpy())), \
                                               str(int(chunk_nb[i].cpu().numpy())), \
                                               str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(']')[-1].split(' ')[1]
            chunk_nb = line.split(']')[-1].split(' ')[2]
            split_nb = line.split(']')[-1].split(' ')[3]
            data = np.fromstring(' '.join(line.split(' ')[1:]).split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    pred_probs = [x[4] for x in ans]  # Get the prediction probabilities for each sample
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    # Calculate precision, recall, AUC value, and confusion matrix
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')

    # Check if label and pred_probs are empty
    # if len(label) > 0 and len(pred_probs) > 0:
    #     auc = roc_auc_score(label, pred_probs, multi_class='ovr')
    # else:
    #     auc = float('nan')  # Set AUC to NaN if empty

    conf_matrix = confusion_matrix(label, pred)

    # Visualize and save the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(eval_path, 'confusion_matrix.png'))

    # Remap classes
    # new_label = [0 if lbl in [0, 1, 2] else 1 for lbl in label]
    new_pred = [0 if prd in [0, 1, 2] else 1 for prd in pred]

    # Calculate new accuracy
    # new_acc = np.mean(np.array(new_label) == np.array(new_pred))
    new_acc = np.mean(np.array(label) == np.array(new_pred))
    # Calculate new AUC value
    if len(label) > 0 and len(pred_probs) > 0:
        # Remap prediction probabilities
        pred_probs = np.array(pred_probs)
        new_pred_probs = pred_probs[:, 3]
        # new_auc = roc_auc_score(new_label, new_pred_probs)
        new_auc = roc_auc_score(label, new_pred_probs)
    else:
        new_auc = float('nan')  # Set AUC to NaN if empty

    # Calculate new confusion matrix
    # new_conf_matrix = confusion_matrix(new_label, new_pred)

    # Visualize and save the new confusion matrix
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(new_conf_matrix, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Double Confusion Matrix')
    # plt.savefig(os.path.join(eval_path, 'double_confusion_matrix.png'))

    # Save sample names, labels, and prediction results to a text file
    with open(os.path.join(eval_path, 'merged_results.txt'), 'w') as f:
        f.write(f'name label pred\n')
        for name, lbl, prd in zip(dict_feats.keys(), label, new_pred):
            f.write(f'{name}\t{lbl}\t{prd}\n')

    return final_top1 * 100, final_top5 * 100, precision * 100, recall * 100, new_auc * 100, new_acc * 100, new_auc * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label), feat]


def merge2(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(']')[-1].split(' ')[1]
            chunk_nb = line.split(']')[-1].split(' ')[2]
            split_nb = line.split(']')[-1].split(' ')[3]
            data = np.fromstring(' '.join(line.split(' ')[1:]).split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    pred_probs = [x[4] for x in ans]  # Get the prediction probabilities for each sample
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    # Calculate precision, recall, AUC value, and confusion matrix
    precision = precision_score(label, pred, average='binary')
    recall = recall_score(label, pred, average='binary')

    pred_probs = np.array(pred_probs)
    # Check if label and pred_probs are empty
    if len(label) > 0 and len(pred_probs) > 0:
        auc = roc_auc_score(label, pred_probs[:, 1])  # Use the second column of prediction probabilities
    else:
        auc = float('nan')  # Set AUC to NaN if empty

    conf_matrix = confusion_matrix(label, pred)

    # Visualize and save the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Double Confusion Matrix')
    plt.savefig(os.path.join(eval_path, 'double_confusion_matrix.png'))
    plt.close()

    # Save sample names, labels, and prediction results to a text file
    with open(os.path.join(eval_path, 'merged_results.txt'), 'w') as f:
        f.write(f'name label pred\n')
        for name, lbl, prd in zip(dict_feats.keys(), label, pred):
            f.write(f'{name}\t{lbl}\t{prd}\n')

    return final_top1 * 100, final_top5 * 100, precision * 100, recall * 100, auc * 100, final_top1 * 100, auc * 100
