import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        # se = torch.mean(x, dim=1, keepdim=True)  # Global average pooling
        se = F.relu(self.fc1(x))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class CombinedModel(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

        # Freeze the weights of the sub-models
        # for param in self.model1.parameters():
        #     param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False

        # # Get the output feature dimensions of the two sub-models
        # self.model1_out_features = self.model1.fc.in_features
        # self.model2_out_features = self.model2.fc.in_features

        # # Remove the classification heads of the sub-models
        # self.model1.fc = nn.Identity()
        # self.model2.fc = nn.Identity()

        # # Define the classification head
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.model1_out_features + self.model2_out_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(1024, 2064)
        # self.fc1_gelu = nn.Sequential(
        #     nn.Linear(2816,512),  # Fully connected layer
        #     nn.GELU(),  # GELU activation function
        #     nn.BatchNorm1d(512, affine=False, eps=1e-6)
        # )
        # self.fc2_gelu = nn.Sequential(
        #     nn.Linear(1024,512),  # Fully connected layer
        #     nn.GELU(),  # GELU activation function
        #     nn.BatchNorm1d(512, affine=False, eps=1e-6)
        # )
        # self.fc_norm1 = nn.LayerNorm(2816)
        # self.fc_norm = nn.LayerNorm(512)
        # self.se1 = SEBlock(2816)
        # self.se = SEBlock(1024)
        self.head = nn.Linear(1024, num_classes)
        self.fc1 = nn.Linear(2816,1024)
        # self.fc2 = nn.Linear(1024,512)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.head = nn.Linear(1024, num_classes)

    def forward(self, x1, x2):
        # Extract features
        features1 = self.model1(x1)
        # features2 = self.model2(x2)
        flag = False
        if x2.size(0) > 3:
        # Split along the first dimension
            split_size = x2.size(0) // 2
            x2 = torch.split(x2, split_size, dim=0)[0]
            flag = True

        outputs = []
        for t in range(x2.shape[2]):
            # Extract the feature vector for the current time step
            if t == 0 or t == 3:
                current_feature = x2[:, :, t, :, :]  # Shape: (bs, 3, 1536, 1536)
                # Input to the encoder
                output = self.model2(current_feature)[-1]  # Shape: (bs, 1024, 48, 48)
                # output = output.to(torch.bfloat16)
                # Append the output to the list
                outputs.append(output)

        # Stack all time step outputs
        outputs = torch.stack(outputs, dim=2)  # Shape: (bs, 1024, 2, 48, 48)

        # Perform averaging
        averaged_output = torch.mean(outputs, dim=2)  # Shape: (bs, 1024, 48, 48)

        # Global average pooling layer

        pooled_feature = self.global_avg_pool(averaged_output)  # Shape: (bs, 1024, 1, 1)
        features2 = pooled_feature.view(pooled_feature.size(0), -1)  # Shape: (bs, 1024)
        

        # Through the sequential model
        # features2 = self.fc_gelu(pooled_feature)  # Shape: (bs, 2064)
        if flag:
            features2 = features2.repeat(2, 1)
        # features2 = features2.repeat(2, 1)  # Shape: (bs, 2064)

        # features1 = self.fc1(features1)
        # combined_features = self.alpha*features1 + (1-self.alpha)*features2

        # Concatenate features
        features1 = self.fc1(features1)  # Shape: (bs, 512)
        # features1 = self.fc_norm(features1)
        # features1 = self.fc1_gelu(features1)
        # features1 = self.se(features1)
        # features2 = self.fc2_gelu(features2)
        # features2 = self.fc_norm(features2)
        # features2 = self.se(features2)

        # features1 = self.fc_norm(features1)
        # features2 = self.fc_norm(features2)

        # combined_features = torch.cat((features1, features2), dim=1)
        combined_features = self.alpha * features1 + (1 - self.alpha) * features2  # Shape: (bs, 1024)
        # combined_features = self.se(combined_features)

        # Classification
        output = self.head(combined_features)
        return output

    def get_num_layers(self):
        return len(self.model1.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed',
            'pos_embed_spatial',
            'pos_embed_temporal',
            'pos_embed_cls',
            'cls_token'
        }
