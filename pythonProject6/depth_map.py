import cv2
import numpy as np
import timm
import torch
import urllib.request
import matplotlib
import tkinter as tk

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class DepthMap:
    def __init__(self, label2):
        self.photo = 'photo.jpg'
        self.label2 = label2
        self.width = 0
        self.height = 0

    def get_map(self):
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        img = cv2.imread(self.photo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        plt.imshow(output)

        plt.axis('off')  # Отключить оси
        plt.tight_layout()  # Установить компактный макет
        plt.savefig('depth_map.png', bbox_inches='tight', pad_inches=0)  # Сохранить изображение

        depth_image = Image.open('depth_map.png')
        self.width, self.height = depth_image.size
        depth_image_tk = ImageTk.PhotoImage(depth_image)

        # Update label2 with the depth map
        self.label2.config(image=depth_image_tk)
        self.label2.image = depth_image_tk

        # plt.imshow(output)
        # plt.show()

