"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import ReLU
import ipdb
import dataset
import util
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
        for module_pos, module in self.model._modules.items():
            if module_pos == 'fc':
                x = x.view(x.shape[0], -1)
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # ipdb.set_trace()
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        weights = np.mean(guided_gradients, axis=(1, 2, 3))  # Take averages for each gradient
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # weights = np.mean(guided_gradients, axis=(2, 3))  # Take averages for each gradient
        # for channel in range(weights.shape[0]):
        #     for depth_idx in range(weights.shape[1]):
        #         cam[depth_idx] += weights[channel][depth_idx] * target[channel][depth_idx]

        # cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0) # RELU
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        # cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask
    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0][0]
        return gradients_as_arr

def reshape_3d_size(img, final_size):
    img = util.resize_3d_img(img, (final_size[1], final_size[2]), normalize=False)
    idx = np.linspace(0, img.shape[0]-2, final_size[0])
    idx = idx.astype(int)
    return img[idx]

if __name__ == '__main__':
    # Get params
    target_example = 2  # Snake
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)
    model = util.load_model('model/ImgType.STRUCTURAL_T1/adam/0.0001/5.ckpt')
    # Grad cam
    grad_cam = GradCam(model, target_layer='conv1')
    d = dataset.ImageDataset('WashU', util.ImgType.STRUCTURAL_FILTER)
    original_image, label, _ = d[15]
    print("label: {}".format(label))
    img = torch.Tensor(original_image).float()
    img = img.view(1, 1, img.shape[0], img.shape[1], img.shape[2]).cuda()
    prep_img = Variable(img, requires_grad=True)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, label)
    guided_backprop = GuidedBackprop(model)
    guided_grads = guided_backprop.generate_gradients(prep_img, label)
    guided_grads = reshape_3d_size(guided_grads, cam.shape)
    reshaped_image = reshape_3d_size(original_image, cam.shape)
    cam_x_original_image = np.multiply(cam, reshaped_image)
    guided_gc = guided_grad_cam(cam, guided_grads)
    display_list = [cam]
    # display_list = [cam, original_image, cam_x_original_image, guided_gc, original_image, guided_grads]
    for d in display_list:
        display = display_img.Display(d)
        display.multi_slice_viewer()
    # display = display_img.Display(cam)
    # display.multi_slice_viewer()
    # display = display_img.Display(guided_gc)
    # display.multi_slice_viewer()
    # display_brain = display_img.Display(original_image)
    # display_brain.multi_slice_viewer()
    # display_brain = display_img.Display(guided_grads)
    # display_brain.multi_slice_viewer()
    print('Grad cam completed')
