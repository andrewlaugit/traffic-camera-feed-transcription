import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageEnhance
import cv2

import os
import glob
import random
from sklearn.utils import shuffle

from pathlib import Path
from app.utils.centroidtracker import *

"""
convert mask to binary mask using threshold
"""


def apply_binary_image_thresholding(mask, threshold=0.5, cuda_allow=False):
    if mask.is_cuda:
        ones_array = torch.ones(mask.shape, device=torch.device('cuda:0'))
        zeros_array = torch.zeros(mask.shape, device=torch.device('cuda:0'))
        threshold_array = ones_array * threshold
    else:
        ones_array = torch.ones(mask.shape)
        zeros_array = torch.zeros(mask.shape)
        threshold_array = ones_array * threshold

    cond = torch.greater(mask, threshold_array)
    mask = torch.where(cond, ones_array, zeros_array)
    torch.cuda.empty_cache()  # Clean up gpu cache
    return mask


"""###View target masks on image
Do a sanity check on data by checking first image and mask target 
"""


def display_masked_image(img, mask, display=True):
    res = img
    res[0] = img[0] + mask[0] * 0.6   # make mask items red
    res[res > 1] = 1

    if res.shape[0] == 1:
        res = np.squeeze(res, axis=0)
    else:
        res = np.moveaxis(res, 0, -1)

    if display:
        plt.imshow(res)
        plt.show()
    return res

# def display_one_image_from_loader(loader):
#     dataiter_train = iter(loader)
#     images, targets = dataiter_train.next()

#     images = images.numpy()[0]
#     masks = targets.numpy()[0]
#     display_masked_image(images, masks)

# display_one_image_from_loader(train_loader)
# display_one_image_from_loader(val_loader)
# display_one_image_from_loader(test_loader)


encode_out = []


def hook(module, input, output):
    encode_out.append(output)


class TrafficCamera_Vgg(nn.Module):
    def __init__(self):
        super(TrafficCamera_Vgg, self).__init__()

        vgg = torchvision.models.vgg.vgg16(pretrained=True)

        # Maxpool output layers
        # NEED TO ADJUST ENCODING OUTPUT LAYERS
        # vgg19 = [4,9,18,27,36]
        # vgg16 = [4,9,16,23,30]
        # vgg11 = [2,5,10,15,20]
        self.encoder_out_layers = [4,9,16,23,30]

        self.vgg = vgg

        # Freeze weights
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # Save intermediate output values
        for i in self.encoder_out_layers:
            self.vgg.features[i].register_forward_hook(hook)

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn_layer1 = nn.BatchNorm2d(512)
        self.deconv_dropout1 = nn.Dropout(p=0.52)

        self.deconv2 = nn.ConvTranspose2d(512+512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn_layer2 = nn.BatchNorm2d(256)
        self.deconv_dropout2 = nn.Dropout(p=0.50)

        self.deconv3 = nn.ConvTranspose2d(256+256, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn_layer3 = nn.BatchNorm2d(128)
        self.deconv_dropout3 = nn.Dropout(p=0.48)

        self.deconv4 = nn.ConvTranspose2d(128+128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn_layer4 = nn.BatchNorm2d(64)
        self.deconv_dropout4 = nn.Dropout(p=0.46)

        self.deconv5 = nn.ConvTranspose2d(64+64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn_layer5 = nn.BatchNorm2d(64)
        self.deconv_dropout5 = nn.Dropout(p=0.44)

        self.deconv6 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1) #, output_padding=1)
        self.deconv_bn_layer6 = nn.BatchNorm2d(3)
        self.deconv_dropout6 = nn.Dropout(p=0.42)

        self.deconv7 = nn.ConvTranspose2d(3+3, 1, 3, stride=1, padding=1)
        self.deconv_bn_layer7 = nn.BatchNorm2d(1)
        self.deconv_dropout7 = nn.Dropout(p=0.4)

    def forward(self, img):
        encode_out.clear()
        out = self.vgg.features(img)

        out = F.relu(self.deconv_bn_layer1(self.deconv1(encode_out[-1])))
        out = self.deconv_dropout1(out)

        out = torch.cat((out, encode_out[-2]), 1)
        out = F.relu(self.deconv_bn_layer2(self.deconv2(out)))
        out = self.deconv_dropout2(out)

        out = torch.cat((out, encode_out[-3]),1)
        out = F.relu(self.deconv_bn_layer3(self.deconv3(out)))
        out = self.deconv_dropout3(out)

        out = torch.cat((out, encode_out[-4]),1)
        out = F.relu(self.deconv_bn_layer4(self.deconv4(out)))
        out = self.deconv_dropout4(out)

        out = torch.cat((out, encode_out[-5]),1)
        out = F.relu(self.deconv_bn_layer5(self.deconv5(out)))
        out = self.deconv_dropout5(out)

        out = F.relu(self.deconv_bn_layer6(self.deconv6(out)))
        out = self.deconv_dropout6(out)

        out = torch.cat((out, img),1)
        out = self.deconv_bn_layer7(self.deconv7(out))
        out = self.deconv_dropout7(out)

        return out


# Load saved model
def load_saved_model():
    model = TrafficCamera_Vgg()
    model_name = 'vgg'
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.001
    model_save_name = 'vgg_augmented_256p_v2_16_25_0.001_best'
    # model_save_name = 'vgg_augmented_16_30_0.001_best'

    models_filepath = Path.cwd() / "models" / model_save_name

    model.load_state_dict(torch.load(models_filepath.__str__()))
    model.eval()

    model = model.cuda()

    return model


def display_one_model_mask(model, loader):
    sigmoid = nn.Sigmoid()

    dataiter_train = iter(loader)
    images, targets = dataiter_train.next()

    if next(model.parameters()).is_cuda:
        images_model = images.cuda()
    out = model(images_model)
    out = sigmoid(out)
    out = apply_binary_image_thresholding(out)

    image = images.numpy()[0]
    targets = targets.numpy()[0]
    mask = out.cpu().detach().numpy()[0]

    display_masked_image(image.copy(), targets.copy())
    display_masked_image(image.copy(), mask.copy())


def find_bounding_boxes(mask, only_bounding_boxes=False):
    mask_orig = mask.copy()
    zeros_array = np.zeros(mask.shape)
    # display_masked_image(mask, zeros_array)

    mask = np.squeeze(mask)
    zeros_array = np.zeros(mask.shape)

    mask = np.uint8(mask * 255)
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if only_bounding_boxes:
       mask = zeros_array.copy()
    idx = 0
    for cnt in contours[0]:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        mask = cv2.rectangle(mask,(x,y),(x+w,y+h),(1),2)
    mask = np.expand_dims(mask, axis=0)
    # display_masked_image(mask_orig, mask, display=False)
    return mask, contours[0]


def display_one_model_mask_with_boxes(model, loader):
    sigmoid = nn.Sigmoid()

    dataiter_train = iter(loader)
    images, targets = dataiter_train.next()

    if next(model.parameters()).is_cuda:
        images_model = images.cuda()
    out = model(images_model)
    out = sigmoid(out)
    out = apply_binary_image_thresholding(out)

    image = images.numpy()[0]
    targets = targets.numpy()[0]
    mask = out.cpu().detach().numpy()[0]
    bound_boxes = find_bounding_boxes(mask.copy())

    truth_plt = display_masked_image(
        image.copy(), targets.copy(), display=False)
    masked_plt = display_masked_image(image.copy(), mask.copy(), display=False)
    bounding_plt = display_masked_image(
        image.copy(), bound_boxes.copy(), display=False)
    bounding_masks_plt = display_masked_image(
        mask.copy(), bound_boxes.copy(), display=False)

    plt.figure(figsize=(24, 10))
    f, axarr = plt.subplots(2, 2, figsize=(12, 5))
    axarr[0, 0].imshow(truth_plt)
    axarr[0, 1].imshow(masked_plt)
    axarr[1, 0].imshow(bounding_plt)
    axarr[1, 1].imshow(bounding_masks_plt)


def transform_image_for_model(image):
    image_transforms1 = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 512)), transforms.ToPILImage()])
    t_image = image_transforms1(image)

    t_image = ImageEnhance.Contrast(t_image).enhance(0.65)
    t_image = ImageEnhance.Brightness(t_image).enhance(1.3)
    t_image = ImageEnhance.Sharpness(t_image).enhance(2)

    image_transform2 = transforms.Compose([transforms.ToTensor()])
    t_image = image_transform2(t_image)

    return t_image


def draw_bounding_boxes_on_image(image,model):
    sigmoid = nn.Sigmoid()
    image_model = torch.unsqueeze(image, axis=0)

    image_model = image_model.cuda()
    out = model(image_model)
    out = sigmoid(out)
    out = apply_binary_image_thresholding(out)

    image = image.numpy()
    mask = out.cpu().detach().numpy()
    bound_boxes, contours = find_bounding_boxes(mask.copy(), only_bounding_boxes=True)
    return display_masked_image(image.copy(), bound_boxes.copy(), display=False), contours


def draw_bounding_boxes_on_image_2(model, ct, frame, t_image, width = 512, height = 256):

    rects = []
    car_past_direction = {}
    # frame = cv2.resize(frame, (width, height))
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image_transforms = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    # t_image = image_transforms(Image.fromarray(img))

    _, contours = draw_bounding_boxes_on_image(t_image, model)
    for cnt in contours:
        # draw a bounding box surrounding the object so we can
        # visualize it
        (startX, startY, endX, endY) = cv2.boundingRect(cnt)
        # if cv2.contourArea(cnt) > 0.6 * endX * endY and
        if cv2.contourArea(cnt) > 150:
            # if cv2.contourArea(cnt) > 50:
            endX = startX + endX
            endY = startY + endY
            if (endX >= 0 and endX <= width and endY >= 0 and endY <= height and startX >= 0 and startX <= width and startY >= 0 and startY <= height):
                rects.append((startX, startY, endX, endY))

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 1)

    objects, object_direction, road_directions = ct.update(rects)
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "Car {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), 1)

        if object_direction[objectID] != 0:
            cv2.arrowedLine(frame, (centroid[0], centroid[1]), (centroid[0]+object_direction[objectID][0], centroid[1]+object_direction[objectID][1]),
                            (255, 0, 0), 1, tipLength=0.5)
            if objectID not in car_past_direction:
                car_past_direction[objectID] = []

        # img_out = np.transpose(img_out, (2, 0, 1))
    # plt.imshow(img_out)
    # plt.show()
    frame = cv2.resize(frame, (width, height))

    total_num_of_car = ct.numberOfObjects()
    totalCarText = "Total # of Cars: " + str(total_num_of_car)
    currentCarText = "Current # of Cars: " + str(len(objects))
    # cv2.putText(frame, totalCarText, (0, 255),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # cv2.putText(frame, currentCarText, (0, 235),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    trafficColor = (0, 255, 0)  # green

    if (len(objects) > 10):  # Red
        trafficColor = (0, 0, 255)
    elif (len(objects) > 7):  # Orange
        trafficColor = (0, 144, 255)

    # Traffic Conditions
    # cv2.putText(frame, "Traffic Condition", (400, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    # cv2.rectangle(frame, (400, 230), (512, 255), trafficColor, -1)

    cv2.rectangle(frame, (390, 180), (512, 256), (255, 255, 255), -1)
    
    if (len(road_directions) > 0):
        cv2.putText(frame, "Direction", (400, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.arrowedLine(frame, (480, 190),
                        (480 + int(road_directions[0][0][0]*15),
                            190 + int(road_directions[0][0][1]*15)),
                        (255, 0, 0), 1, tipLength=0.5)
        # print(road_directions[0][1])
        cv2.putText(frame, str(int(
            road_directions[0][1])), (495, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if (len(road_directions) > 1):
            cv2.putText(frame, "Direction", (400, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            cv2.arrowedLine(frame, (480, 220),
                            (480 + int(road_directions[1][0][0] * 15),
                                220 + int(road_directions[1][0][1] * 15)),
                            (255, 0, 0), 1, tipLength=0.5)
            cv2.putText(frame, str(
                road_directions[1][1]), (495, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if (len(road_directions) > 2):
            cv2.putText(frame, "Direction", (400, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            cv2.arrowedLine(frame, (480, 240),
                            (480 + int(road_directions[2][0][0] * 15),
                                240 + int(road_directions[2][0][1] * 15)),
                            (255, 0, 0), 1, tipLength=0.5)
            # cv2.rectangle(frame, (400, 230), (512, 255), trafficColor, -1)
            cv2.putText(frame, str(
                road_directions[2][1]), (495, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return frame
# # example usage
# image = Image.open(streets_dataset_filepath + '/images/Aptakisic at Bond IP East-9.jpg')
# t_image = transform_image_for_model(image)
# masked_image = draw_bounding_boxes_on_image(t_image)  # this function takes in 1 image and returns the same image with the bounding boxes
# plt.imshow(masked_image)
# plt.show()
