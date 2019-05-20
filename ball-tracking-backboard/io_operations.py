import glob
import os
import cv2
import numpy as np


class IOOperations:
    def __init__(self):
        self.contents = []

    def readParseTxt(self, path):
        temp_array = np.array([])
        with open(path, 'r') as f:
            self.contents = f.readlines()
            # print(self.contents[0])
            for item in self.contents:
                item = item.split("'")
                print(item)
                temp_array = np.append(temp_array, item)

            return temp_array

    def readImage(self, path):
        filenames = glob.glob(path)
        filenames.sort()
        images = [cv2.imread(img) for img in filenames]

        return images
