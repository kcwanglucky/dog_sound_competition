import torch
from torch import nn
from torchsummary import summary
from config import model_config

class AudioClassifier(nn.Module):
    def __init__(self, num_class, config):
        super().__init__()
        conv_layers = []
        last_num_ch = None

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        if config["conv1_channel"]:
            self.conv1 = nn.Conv2d(config["num_channel"], config["conv1_channel"], kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
            self.relu1 = nn.ReLU()
            self.bn1 = nn.BatchNorm2d(config["conv1_channel"])
            torch.nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
            self.conv1.bias.data.zero_()
            conv_layers += [self.conv1, self.relu1, self.bn1]
            last_num_ch = config["conv1_channel"]

        # Second Convolution Block
        if config["conv2_channel"]:
            self.conv2 = nn.Conv2d(config["conv1_channel"], config["conv2_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm2d(config["conv2_channel"])
            torch.nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
            self.conv2.bias.data.zero_()
            conv_layers += [self.conv2, self.relu2, self.bn2]
            last_num_ch = config["conv2_channel"]

        # Third Convolution Block
        if config["conv3_channel"]:
            self.conv3 = nn.Conv2d(config["conv2_channel"], config["conv3_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.relu3 = nn.ReLU()
            self.bn3 = nn.BatchNorm2d(config["conv3_channel"])
            torch.nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
            self.conv3.bias.data.zero_()
            conv_layers += [self.conv3, self.relu3, self.bn3]
            last_num_ch = config["conv3_channel"]

        # Fourth Convolution Block
        if config["conv4_channel"]:
            self.conv4 = nn.Conv2d(config["conv3_channel"], config["conv4_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.relu4 = nn.ReLU()
            self.bn4 = nn.BatchNorm2d(config["conv4_channel"])
            torch.nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
            self.conv4.bias.data.zero_()
            conv_layers += [self.conv4, self.relu4, self.bn4]
            last_num_ch = config["conv4_channel"]

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        # Linear Classifier
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=last_num_ch, out_features=num_class)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.linear(x)

        # Final output
        return x

if __name__ == "__main__":
    data = torch.rand([32, 1, 64, 79])
    model = AudioClassifier(6, model_config)
    pred = model(data)
    # print(model)
    print(summary(model, (1, 64, 79)))
