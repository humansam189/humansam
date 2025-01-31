from engines.engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge, merge2

dir = '/conf_1_fake_sora_best_eval_2/conf_1_fake_sora_best_eval_2_1_fake_sora_image'
final_top1, final_top5, final_precision, final_recall, final_auc, double_acc, double_auc = merge2(dir, 1)

print(
    f"test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%, Precision: {final_precision:.2f}%, Recall: {final_recall:.2f}%, AUC: {final_auc:.2f}%, double ACC: {double_acc:.2f}%, double AUC: {double_auc:.2f}%")

# final_top1, final_top5, final_precision, final_recall, final_auc = merge2(dir, 1)
