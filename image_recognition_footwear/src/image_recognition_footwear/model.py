import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, channel_1, channel_2, channel_3, node_1, node_2, num_classes):
        super().__init__()
        ####### Convolutional layers ######
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channel_1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_1, channel_1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_2),
            nn.LeakyReLU(),
            nn.Conv2d(channel_2, channel_2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_3),
            nn.LeakyReLU(),
            nn.Conv2d(channel_3, channel_3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=7, stride=2),
        )

        ######## Affine layers ########
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_3, node_1),
            nn.BatchNorm1d(node_1),
            nn.Dropout(p=0.5),
            nn.Linear(node_1, node_2),
            nn.BatchNorm1d(node_2),
            nn.Linear(node_2, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        scores = self.fc(x)
        return scores
