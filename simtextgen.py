#@title Dataset Generator
# %% libs
import os
import glob
import time
import requests
from pathlib import Path
import zipfile

import numpy as np
from PIL import ImageFont, ImageDraw, Image

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle

#import torch
#from torch.utils import data

# %% generator
class DatasetChars:
    """Dataset of single char images
    """
    def __init__(
            self,
            n_samples=1000,
            image_size=32, 
            letters=list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            n_objects=1,
            scales=[10,11],
            font_list=['Open Sans'],
            seed=0,
            noise=0.1):
        np.random.seed(seed=seed)
        self.n_classes = len(letters)
        self.classes = {k:letters[k] for k in range(len(letters))}
        self.image_size = image_size
        self.fonts_paths = self.__get_fonts(font_list)
        self.n_objects = n_objects
        self.scales = scales
        self.noise = noise
        self.n_samples = n_samples

        self.list_IDs = list(range(n_samples))
        self.generator = np.random.choice(max(1000000,n_samples*2),n_samples,replace=False)
    
    def __len__(self):
        "Total number of samples"
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        """Generate one sample of data.
        
        Arguments:
            index {id} -- id of the item.
        """
        X, y = self.__create_sample(index)
        return X, y
    
    def __create_sample(self, index):
        if len(self.fonts_paths) == 0:
            return self.generator[index]
        # generate seeds
        np.random.seed(self.generator[index])
        n_objects = np.random.randint(1,self.n_objects+1)
        font_id = np.random.randint(0,len(self.fonts_paths))
        font_id = list(self.fonts_paths.keys())[font_id]
        # create base
        img_empty = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        img_noise = self._get_background(self.image_size,np.random.randint(0,13),self.noise)
        # create font
        fontpath = self.fonts_paths[font_id]
        # draw
        y = []
        for i in range(n_objects):
            class_id = np.random.randint(0,self.n_classes)
            scale = np.random.randint(*self.scales)
            if self.classes[class_id]==" ":
                break
            char = self.__gen_char(
                self.classes[class_id],
                scale,
                fontpath)
            for trydraw in range(3):
                pos = np.random.randint(0,self.image_size-max(char.shape),2)
                if np.sum(img_empty[pos[0]:pos[0]+char.shape[0],pos[1]:pos[1]+char.shape[1],:])==0:
                    img_empty[pos[0]:pos[0]+char.shape[0],pos[1]:pos[1]+char.shape[1],:]=char
                    y.append([
                        class_id,
                        (pos[1]+char.shape[1]*0.5)/img_empty.shape[1],
                        (pos[0]+char.shape[0]*0.5)/img_empty.shape[0],
                        (char.shape[1])/img_empty.shape[1],
                        (char.shape[0])/img_empty.shape[0]])
                    break
        img = img_empty
        if self.noise:
            img_noise[img_empty!=0] = 0
            img = img_noise + img_empty

        # to grayscale
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return img, np.array(y)
    
    @staticmethod
    def __gen_char(text, size, fontpath):
        # generate
        img = Image.fromarray(np.zeros((size*2,size*2*len(text),3), np.uint8))
        font = ImageFont.truetype(fontpath,size)
        draw = ImageDraw.Draw(img)
        draw.text((0,0), text, font=font, fill=(255, 255, 255, 0))
        img = np.array(img)
        # cut
        y0=np.argmax(np.sum(img,axis=(1,2))!=0)
        y1=img.shape[0]-np.argmax(np.flip(np.sum(img,axis=(1,2)))!=0)
        x0=np.argmax(np.sum(img,axis=(0,2))!=0)
        x1=img.shape[1]-np.argmax(np.flip(np.sum(img,axis=(0,2)))!=0)
        return img[y0:y1,x0:x1,:]
    
    @staticmethod
    def plot_labels(X, y, y2=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        # plot image
        if len(X.shape)==3:
            ax.imshow(X)
        else:
            ax.imshow(X, cmap="gray")
        # set ticks
        ax.set_aspect(1)
        ax.set_yticks([i-0.5 for i in range(X.shape[0])], minor='True')
        ax.set_xticks([i-0.5 for i in range(X.shape[1])], minor='True')
        ax.yaxis.grid(True, which='minor')
        ax.xaxis.grid(True, which='minor')
        # plot boxes y2
        boxes=[]
        for i in range(y2.shape[0]):
            boxes.append(Rectangle(
                    (y2[i][1]*X.shape[1]-y2[i][3]*X.shape[1]*0.5-0.5, 
                    y2[i][2]*X.shape[0]-y2[i][4]*X.shape[0]*0.5-0.5), 
                    y2[i][3]*X.shape[1], 
                    y2[i][4]*X.shape[0]
                    ))
        facecolor='b'
        edgecolor='None'
        alpha=0.2
        pc = PatchCollection(boxes, facecolor=facecolor, alpha=alpha,
                            edgecolor=edgecolor)
        ax.add_collection(pc)
        # plot points y2
        boxes=[]
        for i in range(y2.shape[0]):
            boxes.append(Circle(
                    (y2[i][1]*X.shape[1]-0.5, 
                    y2[i][2]*X.shape[0]-0.5),
                    0.5))
        facecolor='b'
        edgecolor='None'
        alpha=0.95
        pc = PatchCollection(boxes, facecolor=facecolor, alpha=alpha,
                            edgecolor=edgecolor)
        ax.add_collection(pc)
        # plot boxes y
        boxes=[]
        for i in range(y.shape[0]):
            boxes.append(Rectangle(
                    (y[i][1]*X.shape[1]-y[i][3]*X.shape[1]*0.5-0.5, 
                    y[i][2]*X.shape[0]-y[i][4]*X.shape[0]*0.5-0.5), 
                    y[i][3]*X.shape[1], 
                    y[i][4]*X.shape[0]
                    ))
        facecolor='g'
        edgecolor='None'
        alpha=0.2
        pc = PatchCollection(boxes, facecolor=facecolor, alpha=alpha,
                            edgecolor=edgecolor)
        ax.add_collection(pc)
        # plot points y
        boxes=[]
        for i in range(y.shape[0]):
            boxes.append(Circle(
                    (y[i][1]*X.shape[1]-0.5, 
                    y[i][2]*X.shape[0]-0.5),
                    0.5))
        facecolor='g'
        edgecolor='None'
        alpha=0.95
        pc = PatchCollection(boxes, facecolor=facecolor, alpha=alpha,
                            edgecolor=edgecolor)
        ax.add_collection(pc)
        return ax
    
    @staticmethod
    def __get_fonts(list_fonts, folder="fonts-cache"):
        #font_styles = ['Black', 'BlackItalic', 'Bold', 'BoldItalic', 'ExtraBold', 'ExtraBoldItalic', 'Light', 'LightItalic', 'Medium', 'MediumItalic', 'Regular', 'RegularItalic', 'SemiBold', 'SemiBoldItalic', 'Thin', 'ThinItalic']
        font_names = dict([(f.replace(" ",""),f) if ("-" in f) else (f.replace(" ","")+"-Regular",f) for f in list_fonts])

        # verify if fonts exist:
        available_fonts = glob.glob(str(Path(folder)/"**"/"*.ttf"),recursive=True)
        available_fonts = {Path(f).stem:f for f in available_fonts}
        non_existing = len(set(font_names.keys()).difference(set(available_fonts.keys())))

        if non_existing:
            # create url
            root_fonts = [f.split("-")[0] for f in list_fonts]
            url = requests.utils.requote_uri("https://fonts.google.com/download?family="+"|".join(root_fonts))
            # download
            for tries in range(3):
                r = requests.get(url)
                if r.status_code==200:
                    break
                time.sleep(1)

            os.makedirs(folder, exist_ok=True)
            temp_zip = str(Path(folder)/'fonts.zip')
            with open(temp_zip, 'wb') as f:
                f.write(r.content)

            #extract
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(folder)
            os.remove(temp_zip)

        # verify success:
        available_fonts = glob.glob(str(Path(folder)/"**"/"*.ttf"),recursive=True)
        available_fonts = {Path(f).stem:f for f in available_fonts}
        non_existing = set(font_names).difference(set(available_fonts.keys()))
        existing = set(font_names).intersection(set(available_fonts.keys()))
        return {font_names[k]:available_fonts[k] for k in existing}
    
    @staticmethod
    def _get_background(size=32, background_type=0, noise=0.1):
        if background_type==0:
            d = np.ones((size,size))*0.5
        elif background_type==1:
            x, y = np.meshgrid(np.linspace(0,1,size), np.linspace(0,0,size))
            d=x
        elif background_type==2:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(0,0,size))
            d=x
        elif background_type==3:
            x, y = np.meshgrid(np.linspace(0,0,size), np.linspace(0,1,size))
            d=y
        elif background_type==4:
            x, y = np.meshgrid(np.linspace(0,0,size), np.linspace(1,0,size))
            d=y
        elif background_type==5:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(1,0,size))
            d = (x+y)/2
        elif background_type==6:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(0,1,size))
            d = (x+y)/2
        elif background_type==7:
            x, y = np.meshgrid(np.linspace(0,1,size), np.linspace(1,0,size))
            d = (x+y)/2
        elif background_type==8:
            x, y = np.meshgrid(np.linspace(0,1,size), np.linspace(0,1,size))
            d = (x+y)/2
        elif background_type==9:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(1,0,size))
            d = np.abs(1-(x+y))/2
        elif background_type==10:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(0,1,size))
            d = np.abs(1-(x+y))/2
        elif background_type==11:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(1,0,size))
            d = (np.abs(0.5-x)+np.abs(0.5-y))
        elif background_type==12:
            x, y = np.meshgrid(np.linspace(1,0,size), np.linspace(0,1,size))
            d = 1-(np.abs(0.5-x)+np.abs(0.5-y))
        d = (d - np.random.uniform(-noise/2,noise/2,(size, size)))*250
        d = np.clip(d,0,240)
        base = np.stack([d,d,d],axis=2)
        return base.astype(np.uint8)

if __name__ == "__main__":
    # example multiple
    fontlist = ["Roboto-Bold", "Open Sans-BoldItalic", "Livvic-Black Italic"]
    d = DatasetChars(
        font_list=fontlist,
        letters=list(" 0123456789"),
        n_objects=2,
        image_size=32,
        scales=[20,21],
        noise=0.5)
    for i in range(3):
        X, y = d[i]
        d.plot_labels(X,y,y2=y)
        plt.show()