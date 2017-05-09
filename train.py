# -*- coding: utf-8 -*-
# file: train.py
# author: JinTian
# time: 09/05/2017 2:11 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.image as mimage
import copy
from models import get_input_param_optimizer, get_style_model_and_losses
import sys

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

image_size = 512 if use_cuda else 512  # use small size if no gpu


class ImageUtil(object):
    """
    this class load an image from file path,
    return a torch tensor
    """
    def __init__(self):
        self.loader = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor()
        ])
        self.un_loader = transforms.ToPILImage()

    def load_image(self, image_name):
        image = Image.open(image_name)
        image = Variable(self.loader(image))
        image = image.unsqueeze(0)
        return image

    def show_image(self, tensor, title=None):
        """
        this method will convert
        :return:
        """
        image = tensor.clone().cpu()
        image = image.view(3, image_size, image_size)
        image = self.un_loader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.show()

    def save_image(self, tensor, name='default.jpg'):
        image = tensor.clone().cpu()
        image = image.view(3, image_size, image_size)
        image = self.un_loader(image)
        print(image)
        try:
            im = Image.fromarray(image)
            im.save(name)
            print('image saved.')
        except Exception as e:
            print(e)
            mimage.imsave(name, image)
            print('image saved.')


def train():
    """
    train process of style transfer net.
    :return:
    """
    image_util = ImageUtil()

    image_1 = "images/picasso.jpg"
    image_2 = "images/dancing.jpg"
    if len(sys.argv) > 2:
        image_1 = sys.argv[1]
        image_2 = sys.argv[2]
    print('image 1: {}, image 2: {}'.format(image_1, image_2))
    style_img = image_util.load_image(image_1).type(dtype)
    content_img = image_util.load_image(image_2).type(dtype)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    # plt.figure()
    # image_util.show_image(style_img.data, title='Style Image')
    #
    # plt.figure()
    # image_util.show_image(content_img.data, title='Content Image')

    style_weight = 1000
    content_weight = 3
    num_steps = 900

    input_img = content_img.clone()
    model, style_losses, content_losses = get_style_model_and_losses(style_img, content_img, style_weight,
                                                                     content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + style_score

        optimizer.step(closure)
    input_param.data.clamp_(0, 1)
    output = input_param.data
    image_util.save_image(output, name='output2.jpg')


def main():
    train()


if __name__ == '__main__':
    main()