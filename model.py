import torch
import torch.nn as nn
import torchvision.models
OUTPUT_CLASSES = 4

IMAGE_SHAPE = (200, 200)
def max_pool_output_shape(input_shape, max_poolsize):
    res = []
    for i, p in zip(input_shape, max_poolsize):
        res.append(i-(p-1))
    return tuple(res)

def _conv_layer(input_channels, output_channels, kernel_size):
    return torch.nn.Sequential(nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size),
                               nn.ReLU())

# training a 3D conv net on the structural dataset
class StructuralModel3D(nn.modules.Module):
    def __init__(self):
        super(StructuralModel3D, self).__init__()
        max_pool_shape = (2,2,2)
        self.max_pool_1 = torch.nn.MaxPool3d(max_pool_shape)
        # img_shape = (input_length, *IMAGE_SHAPE)
        # max_pool_output = max_pool_output_shape(img_shape, max_pool_shape)
        self.conv1 = _conv_layer(1, 32, kernel_size=5)
        self.max_pool_2 = torch.nn.MaxPool3d(max_pool_shape)
        self.conv2 = _conv_layer(32, 32, kernel_size=(3, 5, 5))
        self.conv3 = _conv_layer(32, 64, kernel_size=(3,3,3))
        self.conv4 = _conv_layer(64, 64, kernel_size=3)
        self.fc = torch.nn.Linear(64, 4)

    def forward(self, img):
        out = self.max_pool_1(img)
        out = self.conv1(out)
        out = self.max_pool_2(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if len(img.shape) == 5:
            out = out.view(img.shape[0], -1)
        else:
            out = out.view(1, -1)
        out = self.fc(out)
        return out
