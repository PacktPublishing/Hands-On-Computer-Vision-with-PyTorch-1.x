from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
fig, plt_arr = plt.subplots(2, 3)
((style_plt, content_plt, stylized_plt), (style_loss_plt, content_loss_plt, total_loss_plt)) = plt_arr

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) # add batch dimension
    return image.to(device, torch.float)

def plot_image(tensor, fig, title=None):
    image = tensor.cpu().clone() 
    image = image.squeeze(0)      
    image = unloader(image)
    fig.imshow(image)
    fig.axis('off')
    if title is not None:
        fig.title.set_text(title)
    plt.pause(0.001) # pause a bit so that plots are updated

style_img = image_loader("./images/style-images/mosaic.jpg")
content_img = image_loader("./images/content-images/amber.jpg")
input_img = content_img.clone()

plot_image(style_img, style_plt, title='Style Image')
plot_image(content_img, content_plt, title='Content Image')
plot_image(input_img, stylized_plt, title='Stylized Image 0')


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    batch, channels, height, width = input.size()  

    features = input.view(batch * channels, height * width) 

    G = torch.mm(features, features.t())  

    return G.div(batch * channels * height * width)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(style_img, content_img)

    optimizer = optim.SGD([input_img.requires_grad_()], lr=0.001)

    style_losses_plt = []
    content_losses_plt = []
    total_losses_plt = []

    print('Optimizing..')
    
    for it in range(num_steps):
        # correct the values of updated input image
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()
        optimizer.step()

        style_losses_plt.append(style_score)
        content_losses_plt.append(content_score)
        total_losses_plt.append(style_score + content_score)
        
        if it % 50 == 0:
            print('Iteration: %i, Style Loss : %.3f Content Loss: %.3f' % \
                (it, style_score.item(), content_score.item()))
            style_loss_plt.cla()
            style_loss_plt.plot(style_losses_plt, color='r')
            style_loss_plt.title.set_text('Style Loss')
            content_loss_plt.cla()
            content_loss_plt.plot(content_losses_plt,  color='g')
            content_loss_plt.title.set_text('Content Loss')
            total_loss_plt.cla()
            total_loss_plt.plot(total_losses_plt,  color='b')
            total_loss_plt.title.set_text('Total Loss')
            plot_image(input_img, stylized_plt, title='Stylized Image %i' % it)

    plt.pause(100)

run_style_transfer(content_img, style_img, input_img)
