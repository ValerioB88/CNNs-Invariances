import re
import torch
import os
import pathlib
import glob
from PIL import Image, ImageEnhance
import numpy as np
from torchvision import transforms
folder = './data/ETH-80-master'
output = folder + '/transformed_darkened_excluding_extreme_views'
# output = folder + '/only_resized/'
class_list = range(11)
obj_list = range(11)

for class_num in class_list:
    for object_num in obj_list:
        all_images = glob.glob(folder + '/original/' + str(class_num) + '/' + str(object_num) + '/**.png')
        for img in all_images:

            basename = os.path.splitext(os.path.basename(img))[0]
            class_name = re.findall(r'[a-zA-Z]+', basename)[0]
            pathlib.Path(f'{output}/{class_name}/').mkdir(parents=True, exist_ok=True)

            g = re.search(r'-(\d+)-(\d+)', basename).groups()
            incl, azi = int(g[0]), int(g[1])

            map = f'{folder}/original/{class_num}/{object_num}/maps/{basename}-map.png'

            # exclude extreme views (in practice, it's only a couple of views per object)
            # and it works more or less the same without excluding them.
            if incl < 30 or incl > 115:
                continue
            # apply map and convert to grayscale without changing channel num
            img_pil = np.array(Image.open(img))
            map_pil = Image.open(map)
            c = np.array(map_pil)
            img_pil[np.where(c[:, :, 1] != 255)] = 0
            img_pil = Image.fromarray(img_pil)
            a = transforms.Resize(size=(128, 128))(transforms.Grayscale()(img_pil))
            # a = transforms.Resize(size=(128, 128))(img_pil)
            rgbimg = Image.new("RGB", a.size)
            rgbimg.paste(a)
            rgbimg = ImageEnhance.Brightness(rgbimg).enhance(0.7)
            rgbimg.save(f'{output}/{class_name}/O{object_num}_I{incl}_A{azi}.png')

