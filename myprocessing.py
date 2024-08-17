from predictor import Predictor, SingleNail
from utils import *
import cv2
import random as rng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# variabili da impostare
SOGLIA_AREA = 100
ITER_ERODE = 5
SOGLIA_REGIONE_UNGHIA = 0.95

class MyProcessing:
    def __init__(self):
        pass

    def riduzione_morfologica(self, mask: np.ndarray, iters = ITER_ERODE, threshold=SOGLIA_REGIONE_UNGHIA):
        tmp = mask.copy()
        threshold = float(threshold)
        tmp[tmp <= threshold] = 0
        tmp[tmp > threshold] = 1
        tmp = tmp.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(tmp, kernel, iterations=iters)
        return erosion

    def contorni(self, mask: np.ndarray, base, threshold=100):
        src_gray = mask.copy()
        src_gray = src_gray.astype(np.uint8)
        src_gray = src_gray * 255
        # Detect edges using Canny
        canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
        # plt.figure()
        # plt.imshow(canny_output)
        # plt.show()
        # Find contours
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundinrrect = []
        for item in contours:
            boundinrrect.append( cv2.boundingRect(item) )

        # Draw contours
        # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        drawing = base.copy()
        i = 0
        masks = []
        centroids = []
        areas = []
        for cnt in contours:
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, contours, i, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)
            mu = cv2.moments(cnt)
            mc = (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))
            # print(mc)
            # print(f'index {i}')
            # send area data
            areas.append(cv2.contourArea(cnt))
            centroids.append(mc)
            # create mask for blobs
            tmp = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(tmp, contours, i, (255, 255, 255), cv2.FILLED)
            masks.append(tmp)
            i = i + 1

        return masks, centroids, areas, drawing, boundinrrect

    def apply(self, nn_threshold = None, morpho_iter = None, force = False):

        if not hasattr(self, "unghie") or force:
            class_dict_path = get_final_path(0, ['labels', 'label_class_dict.csv'])
            print(f'>>>> {class_dict_path}')
            class_dict = pd.read_csv(class_dict_path)
            class_names = class_dict['name'].tolist()
            class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
            select_classes = ['background', 'nail']
            select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
            select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

            predictor = Predictor("current.jpg", select_class_rgb_values, select_classes, device="cpu")
            processed_img = predictor.img_preprocess()
            self.processing_img = processed_img
            self.mask = predictor.get_predicted_mask(processed_img)
            self.unghie = self.mask[:, :, select_classes.index('nail')]
            # predictor.get_visualize(self.mask)

            #first call with new image
            image_vis = SingleNail("current.jpg", augmentation=get_validation_augmentation(),
                                   class_rgb_values=select_class_rgb_values)[0]
            image_vis = Predictor.crop_image(image_vis.astype('uint8'))
            self.base_image = Image.fromarray(image_vis, 'RGB')

        unghie = self.unghie.copy()

        # morhological operator
        base_image = np.asarray(self.base_image)
        if morpho_iter is None:
            morpho_iter = ITER_ERODE
        mask = self.riduzione_morfologica(unghie, iters=morpho_iter, threshold=nn_threshold)
        filtermask, centroids, areas, drawing, boundinrrect = self.contorni(mask, base_image)

        labels = []
        for layer in filtermask:
            # plt.imshow(layer)
            # plt.show()
            res = cv2.mean(base_image, layer[:, :, 0])
            res = np.asarray(res)
            res = np.round(res * 10) / 10
            labels.append(res[:-1])

        return drawing, centroids, labels, boundinrrect, base_image

    def get_progessing_img(self):
        return self.processing_img