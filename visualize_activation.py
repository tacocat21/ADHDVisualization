"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
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
        negative_cam = np.ones(target.shape[1:], dtype=np.float32)
        weights = np.mean(guided_gradients, axis=(1, 2, 3))  # Take averages for each gradient
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            negative_cam -= w * target[i, :, :]

        # cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0) # RELU
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1

        negative_cam = np.maximum(cam, 0) # RELU
        negative_cam = (negative_cam - np.min(negative_cam)) / (np.max(negative_cam) - np.min(negative_cam))  # Normalize between 0-1
        beta = 1000
        return beta*cam, beta*negative_cam

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
        first_layer = list(self.model._modules.items())[1][1][0]
        print(first_layer)
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

def upsize_3d_img(img, final_size):
    # resize x, y
    resize_x_y = []
    for i in range(img.shape[0]):
        resize_x_y.append(cv2.resize(img[i], (final_size[1], final_size[2])).transpose())
    resize_x_y = np.asarray(resize_x_y)
    ipdb.set_trace()
    resize_x_y = resize_x_y.transpose(1, 0, 2)

    final = []
    for i in range(resize_x_y.shape[0]):
        final.append(cv2.resize(resize_x_y[i], (final_size[0], final_size[2])).transpose())
    final = np.asarray(final)
    final = final.transpose(1, 0, 2)

    return final

def create_images(model, model_name, dataset_name='Peking_1', img_idx=2, target_layer='conv1', image_dir = 'images/'):
    # Grad cam
    grad_cam = GradCam(model, target_layer=target_layer)
    d = dataset.ImageDataset(dataset_name, util.ImgType.STRUCTURAL_FILTER)
    original_image, label, _ = d[img_idx]
    print("label: {}".format(label))
    img = torch.Tensor(original_image).float()
    img = img.view(1, 1, img.shape[0], img.shape[1], img.shape[2]).cuda()
    prep_img = Variable(img, requires_grad=True)
    # Generate cam mask
    cam, negative_cam = grad_cam.generate_cam(prep_img, label)
    guided_backprop = GuidedBackprop(model)
    guided_grads = guided_backprop.generate_gradients(prep_img, label)
    # cam = upsize_3d_img(cam, guided_grads.shape)
    guided_grads_small = reshape_3d_size(guided_grads, cam.shape)
    original_image_small = reshape_3d_size(original_image, cam.shape)
    # reshaped_image = reshape_3d_size(original_image, cam.shape)
    # cam_x_original_image = np.multiply(cam, reshaped_image)
    guided_gc = guided_grad_cam(cam, guided_grads_small)
    layer_name_dict = target_layer[-1]
    display_list = []
    label_name = ['Normal patient', 'Patient with ADHD-Combined', 'Patient with ADHD-hyperactive/Impulsive', 'Patient with ADHD-Inattentive']
    # negative_cam, cam, guided_gc, original_image, original_image, guided_grads
    display_list.append({'img': negative_cam, 'title': 'Negative Grad-CAM for Epoch {}. {}'.format(model_name, label_name[int(label)]), 'filename': 'negative_cam.png'})
    display_list.append({'img': cam, 'title': 'Grad-CAM for Epoch {} Convolution layer {} {}'.format(model_name, target_layer[-1], label_name[int(label)]), 'filename': 'cam.png'})
    display_list.append({'img': original_image, 'title': 'Original Structural MRI Scan.\nLabel: {}'.format(label_name[int(label)]), 'filename': 'original_image.png'})
    display_list.append({'img': guided_gc, 'title': 'Guided Grad-CAM for Epoch {} Conv layer {} {}'.format(model_name, target_layer[-1], label_name[int(label)]), 'filename': 'guided_gc.png'})
    display_list.append({'img': guided_grads, 'title': 'Guided Backpropagation for Epoch {} {}'.format(model_name, label_name[int(label)]), 'filename': 'guided_backprop.png'})

    util.mkdir(image_dir)
    for d in display_list:
        display_img.save_3d_image_display(d['img'], d['title'], os.path.join(image_dir, d['filename']))


if __name__ == '__main__':
    # model = torch.load('model/ImgType.STRUCTURAL_T1/adam/0.0001/25.ckpt')
    # model = torch.load('model_large/ImgType.STRUCTURAL_T1/adam/0.0001/149.ckpt')
    # dir_name = 'model_large/ImgType.STRUCTURAL_T1/adam/0.001/'
    # models = [19, 25, 29, 75, 100, 125, 149]
    # for m in models:
    #     model = torch.load(dir_name + '{}.ckpt'.format(m))
    #     # model = util.load_model(dir_name + '{}.ckpt'.format(m))
    #     create_images(model, target_layer='conv1', model_name=str(m), image_dir='images/large/0.001/conv1/{}/'.format(m))
    #     create_images(model, target_layer='conv2', model_name=str(m), image_dir='images/large/0.001/conv2/{}/'.format(m))
    #     create_images(model, target_layer='conv4', model_name=str(m), image_dir='images/large/0.001/conv4/{}/'.format(m))
    #     create_images(model, target_layer='conv8', model_name=str(m), image_dir='images/large/0.001/conv8/{}/'.format(m))

    dir_name = 'model/ImgType.STRUCTURAL_T1/adam/0.0001/'
    models = [22]
    for m in models:
        model = torch.load(dir_name + '{}.ckpt'.format(m))
        # model = util.load_model(dir_name + '{}.ckpt'.format(m))
        create_images(model, img_idx=0, target_layer='conv1', model_name=str(m), image_dir='images/small/0.0001/conv1/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv2', model_name=str(m), image_dir='images/small/0.0001/conv2/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv3', model_name=str(m), image_dir='images/small/0.0001/conv3/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv4', model_name=str(m), image_dir='images/small/0.0001/conv4/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv5', model_name=str(m), image_dir='images/small/0.0001/conv5/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv6', model_name=str(m), image_dir='images/small/0.0001/conv6/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv7', model_name=str(m), image_dir='images/small/0.0001/conv7/{}/'.format(m))
        create_images(model, img_idx=0, target_layer='conv8', model_name=str(m), image_dir='images/small/0.0001/conv8/{}/'.format(m))
