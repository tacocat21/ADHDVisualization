"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import model as _m
import cv2
import numpy as np
import torch
import util
import dataset
from torch.autograd import Variable
import ipdb
import display_img
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """

        conv_output = None
        count = 0
        for module in self.model.children():
            if type(module) == torch.nn.Linear:
                x = x.view(x.shape[0], -1)
            if type(module) == torch.nn.Sequential:
                for child in module.children():
                    x = child(x)  # Forward
                    if count == self.target_layer:
                        x.register_hook(self.save_gradient)
                        conv_output = x  # Save the convolution output on that layer
                    count += 1
            else:
                x = module(x)
                if count == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                count += 1
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model.cuda()
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        print(input_image.shape)
        # original_shape = (input_image.shape[2], input_image.shape[3], input_image.shape[4])
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.cuda()
        # Zero grads
        self.model.zero_grad()
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        # ipdb.set_trace()
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        res = []
        for i, w in enumerate(weights):
            cam += w * target[i, :, :, :]
        for i in range(len(cam)):
            res.append(cv2.resize(cam[i], util.MODEL_IMG_INPUT_SIZE))
        cam = np.asarray(res).transpose(0, 2, 1)
        # cam = np.asarray(cam)
        # cam = np.resize(cam, original_shape)
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam

def reshape_3d_size(img, final_size):
    idx = np.linspace(0, img.shape[0]-2, final_size[0])
    idx = idx.astype(int)

    return img[idx]

def gen(img, label, model):
    out, conv = model.forward(img, get_conv8=True)
    one_hot_output = torch.FloatTensor(1, out.size()[-1]).zero_().cuda()
    one_hot_output[0][label] = 1
    model.zero_grad()
    back = out.backward(gradient=one_hot_output, retain_graph=True, create_graph=True)
    ipdb.set_trace()
    print(back)

def run():
    # Get params
    target_example = 2  # Snake
    d = dataset.ImageDataset('Peking_1', img_type=util.ImgType.STRUCTURAL_T1)
    idx = 1
    original_img = d[idx][0]
    img = torch.Tensor(original_img).float()
    img = img.view(1, 1, img.shape[0], img.shape[1], img.shape[2]).cuda()
    img = Variable(img, requires_grad=True)
    label = d[idx][1]
    print(label)
    _model = _m.StructuralModel3DFullImage()
    old_model = torch.load('model/ImgType.STRUCTURAL_FILTER/adam/0.0001/29.ckpt')
    _model.load_state_dict(old_model.state_dict())
    _model = _model.cuda()
    for m in _model.children():
        m.requires_grad = True
    # gen(img, label, _model)
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(_model, target_layer=2)
    # Generate cam mask
    cam = grad_cam.generate_cam(img, int(d[1][1]))
    print(cam.shape)
    # ipdb.set_trace()

    original_img = np.asarray(original_img)
    original_img = reshape_3d_size(original_img, cam.shape)
    # d2 = display_img.Display(original_img, mask=cam)
    # d2.multi_slice_viewer()

    display = display_img.Display(cam)
    display.multi_slice_viewer()
    # Save mask
    # save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')

if __name__ == '__main__':
    run()