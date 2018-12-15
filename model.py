import torch
import torch.nn as nn
import ipdb
import torchvision.models
OUTPUT_CLASSES = 4

IMAGE_SHAPE = (200, 200)
def max_pool_output_shape(input_shape, max_poolsize):
    res = []
    for i, p in zip(input_shape, max_poolsize):
        res.append(i-(p-1))
    return tuple(res)

def _conv_layer(input_channels, output_channels, kernel_size, stride=1, padding=0):
    return torch.nn.Sequential(nn.Conv3d(input_channels, output_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm3d(output_channels))

class StructuralModel3DFullImageLarge(nn.modules.Module):
    def __init__(self):
        super(StructuralModel3DFullImageLarge, self).__init__()
        max_pool_shape = (2,2,2)
        self.max_pool_1 = torch.nn.MaxPool3d(max_pool_shape)
        # img_shape = (input_length, *IMAGE_SHAPE)
        # max_pool_output = max_pool_output_shape(img_shape, max_pool_shape)
        self.conv1 = _conv_layer(1, 64, kernel_size=5)
        self.max_pool_2 = torch.nn.MaxPool3d(max_pool_shape)
        self.conv2 = _conv_layer(64, 64, kernel_size=(4, 5, 4), stride=(2, 2, 2))
        self.conv3 = _conv_layer(64, 48, kernel_size=(4, 4, 4))
        self.conv4 = _conv_layer(48, 32, kernel_size=(4, 4, 4))
        self.conv5 = _conv_layer(32, 32, kernel_size=(4, 4, 4))
        self.conv6 = _conv_layer(32, 32, kernel_size=(4, 3, 4))
        self.conv7 = _conv_layer(32, 32, kernel_size=(4, 3, 3))
        self.conv8 = _conv_layer(32, 16, kernel_size=(3, 3, 3))
        self.conv9 = _conv_layer(16, 16, kernel_size=(3, 3, 3))

        # self.conv8[0].register_hook(self.save_gradient)
        self.fc = torch.nn.Linear(128, 4)
        self.gradient = None

    def save_gradient(self, grad):
        print(grad)
        self.gradient= grad

    def forward(self, img, get_conv9=False):
        out = self.max_pool_1(img)
        out = self.conv1(out)
        out = self.max_pool_2(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)

        if len(img.shape) == 5:
            out = out.view(img.shape[0], -1)
        else:
            out = out.view(1, -1)
        out = self.fc(out)
        return out


class StructuralModel3DFullImage(nn.modules.Module):
    def __init__(self):
        super(StructuralModel3DFullImage, self).__init__()
        max_pool_shape = (2,2,2)
        self.max_pool_1 = torch.nn.MaxPool3d(max_pool_shape)
        # img_shape = (input_length, *IMAGE_SHAPE)
        # max_pool_output = max_pool_output_shape(img_shape, max_pool_shape)
        self.conv1 = _conv_layer(1, 32, kernel_size=5)
        self.max_pool_2 = torch.nn.MaxPool3d(max_pool_shape)
        self.conv2 = _conv_layer(32, 32, kernel_size=(4, 5, 4), stride=(2, 2, 2))
        self.conv3 = _conv_layer(32, 64, kernel_size=(4, 4, 4))
        self.conv4 = _conv_layer(64, 64, kernel_size=(4, 4, 4))
        self.conv5 = _conv_layer(64, 32, kernel_size=(4, 4, 4))
        self.conv6 = _conv_layer(32, 16, kernel_size=(4, 4, 4))
        self.conv7 = _conv_layer(16, 8, kernel_size=(3, 4, 3))
        self.conv8 = _conv_layer(8, 4, kernel_size=(3, 3, 3))
        # self.conv8[0].register_hook(self.save_gradient)
        self.fc = torch.nn.Linear(120, 4)
        self.gradient = None

    def save_gradient(self, grad):
        print(grad)
        self.gradient= grad

    def forward(self, img, get_conv8=False):
        out = self.max_pool_1(img)
        out = self.conv1(out)
        out = self.max_pool_2(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        if not get_conv8:
            out = self.conv8(out)
        else:
            for i in range(len(self.conv8)):
                m = self.conv8[i]
                out = m(out)
                if type(m) == nn.Conv3d:
                    conv_out = out

        if len(img.shape) == 5:
            out = out.view(img.shape[0], -1)
        else:
            out = out.view(1, -1)
        # ipdb.set_trace()
        out = self.fc(out)
        if get_conv8:
            return out, conv_out
        return out

