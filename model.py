import torch
from torch import nn
from torchsummary import summary
from config import model_config

class AudioTransformer(nn.Module):
    def __init__(self, num_class, config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.transformer_encoder(x)

        return 

class AudioClassifier(nn.Module):
    def __init__(self, num_class, config):
        super().__init__()
        conv_layers = []
        last_num_ch = None

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        if config["conv1_channel"]:
            self.conv1 = nn.Conv2d(config["num_channel"], config["conv1_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))
            self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu1 = nn.ReLU()
            self.bn1 = nn.BatchNorm2d(config["conv1_channel"])
            # self.ln1 = nn.LayerNorm([16, 16, 51], elementwise_affine=False)
            # self.dropout1 = nn.Dropout(0.1)
            # torch.nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
            # self.conv1.bias.data.zero_()
            # conv_layers += [self.conv1, self.mp1, self.relu1, self.bn1]
            conv_layers += [self.conv1, self.relu1, self.bn1]
            last_num_ch = config["conv1_channel"]

        # Second Convolution Block
        if config["conv2_channel"]:
            self.conv2 = nn.Conv2d(config["conv1_channel"], config["conv2_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm2d(config["conv2_channel"])
            # self.ln2 = nn.LayerNorm([32, 4, 13], elementwise_affine=False)
            # self.dropout2 = nn.Dropout(0.1)
            # torch.nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
            # self.conv2.bias.data.zero_()
            conv_layers += [self.conv2, self.relu2, self.bn2]
            last_num_ch = config["conv2_channel"]

        # Third Convolution Block
        if config["conv3_channel"]:
            self.conv3 = nn.Conv2d(config["conv2_channel"], config["conv3_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu3 = nn.ReLU()
            self.bn3 = nn.BatchNorm2d(config["conv3_channel"])
            # torch.nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
            # self.conv3.bias.data.zero_()
            # conv_layers += [self.conv3, self.relu3, self.bn3]
            conv_layers += [self.conv3, self.relu3, self.bn3]
            last_num_ch = config["conv3_channel"]

        # Fourth Convolution Block
        if config["conv4_channel"]:
            self.conv4 = nn.Conv2d(config["conv3_channel"], config["conv4_channel"], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu4 = nn.ReLU()
            self.bn4 = nn.BatchNorm2d(config["conv4_channel"])
            torch.nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
            self.conv4.bias.data.zero_()
            conv_layers += [self.conv4, self.mp4, self.relu4, self.bn4]
            last_num_ch = config["conv4_channel"]

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        # Linear Classifier
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=last_num_ch, out_features=num_class)
        # self.linear1 = nn.Linear(in_features=last_num_ch, out_features=16)
        # self.relu5 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.1)

        # self.linear2 = nn.Linear(in_features=16, out_features=num_class)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.linear(x)
        # x = self.linear1(x)
        # self.relu5(x)
        # self.dropout1(x)
        # x = self.linear2(x)

        # Final output
        return x

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = nn.functional.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class AudioClassifierCNNGRU(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(AudioClassifierCNNGRU, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.linears = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(101 * 16, 32)
        self.linear2 = nn.Linear(32, n_class)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.linears(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x

if __name__ == "__main__":
    data = torch.rand([32, 1, 64, 201])
    model = AudioClassifier(6, model_config)
    # model = AudioClassifierCNNGRU(n_cnn_layers=2, n_rnn_layers=1, rnn_dim=16, n_class=6, n_feats=64)
    # model = AudioTransformer(6, model_config)

    pred = model(data)
    print(pred)
    print(summary(model, (1, 64, 201)))
