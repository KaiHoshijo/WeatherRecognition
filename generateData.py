# created by Kai Hoshijo
# date: 3/28/2019
# time: 10:35 AM
# mood: not tired

#%%
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image as image
#%%
class generateData:
    def __init__(self, img_size):
        # self.path = self.getPath()
        self.path = r"C:\Users\kaiho\.VirtualBox\WeatherRecognition\dataset2"
        self.img_size = img_size
        self.x, self.y = self.organiseData()
        for x in range(len(self.x)-1):
                if self.x[x].shape == self.x[x+1].shape:
                    continue
                else:
                    print("there is a problem with self.x[{}] with shape {}".format(
                        x+1, self.x[x+1].shape))


    def organiseData(self):
        dataNames = os.listdir(self.path)
        print("Different data set names:", " ".join(dataNames))
        one_hot = [[0 for i in range(len(dataNames))] for x in dataNames]
        for name in range(len(dataNames)):
            one_hot[name][name] = 1
        # df_dict = {}
        x_list = []
        y_list = []
        for name in range(len(dataNames)):
            # temp_list = []
            for pic_name in os.listdir(os.path.join(self.path, dataNames[name])):
                temp_path = os.path.join(self.path, "{}\\{}".format(dataNames[name], pic_name))
                # resize_image 
                temp_img = image.open(temp_path)
                if len(np.asarray(temp_img).shape) == 3:
                        if np.asarray(temp_img).shape[2] == 3:
                            # temp_list.append(temp_img)
                            # df_dict[temp_img] = name
                            smaller_img = temp_img.resize((self.img_size, self.img_size))
                            # smaller_img = temp_img
                            x_list.append(np.asarray(smaller_img) / 255.0)
                            y_list.append(one_hot[name])
                else:
                    continue

        print(dataNames, " = ", one_hot)
        return np.asarray(x_list), np.asarray(y_list)
    
    def splitData(self, testRatio, random_state):
        return train_test_split(self.x, self.y, test_size = testRatio, random_state = random_state)

    def get_size(self):
        heights = []
        widths = []
        for img in self.x:
            heights.append(img.shape[0])
            widths.append(img.shape[1])
        avg_height = sum(heights) / len(heights)
        avg_width = sum(widths) / len(widths)
        print("Max height:", max(heights))
        print("Min height:", min(heights))
        print("Average height:", avg_height)
        print("Max width:", max(widths))
        print("Min width:", min(widths))
        print("Average width:", avg_width)

# #%%
# generate = generateData()
# generate.show_image(0)
    
