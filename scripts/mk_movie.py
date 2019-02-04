from os import path, remove, walk #, listdir
import numpy as np
import cv2
from scipy.misc import imresize as imresize
import rasterio
from imageio import mimwrite

root_dir = '/Users/roserustowicz/Desktop/data_examples/scene_000065/s2'

for root, dirs, files in walk(root_dir):
    fnames = []

    for f in files:
        if f.endswith(".tif") and 's2' in root_dir:
            fnames.append(path.join(root_dir, f))
        elif f.endswith("_analytic.tif"):
            fnames.append(path.join(root_dir, f))

fnames.sort()
img_list = []
for idx in range(0, len(fnames)):
    print(fnames[idx])

    with rasterio.open(fnames[idx]) as src:
        img = src.read()

    if np.sum(img[0, :, :] == 0)/len(img[0, :, :]) > 0.5:
        pass
    else:
        if 's2' in fnames[idx]:
            date = fnames[idx].split('/')[-1].split('_')[-1].replace('.tif', '').replace('2017-', '').replace('-', '')
            img = img[:3, :, :]
            img[img > 2000] = 2000
            img[img < 1000] = 1000
        else:
            date = fnames[idx].split('/')[-1].split('_')[5]
            img = img[:3, :211, :211]
            img[img > 8000] = 8000
            img[img < 5000] = 5000

        img = np.transpose(img, (1, 2, 0))
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
        img = img.astype(np.uint8)
        img = cv2.putText(img=img.copy(), text=date, org=(0, int(img.shape[1]-5)) ,fontFace=1, fontScale=1, color=(255,0,0), thickness=1)

        img_list.append(img)

if len(img_list) > 0: # and len(vh_img_list) > 0:
    img_array = np.concatenate([arr[np.newaxis] for arr in img_list])

if 's2' in root_dir:
    mimwrite(root_dir + 'video_s2.mp4', img_array, fps=2)
else:
    mimwrite(root_dir + 'video_planet.mp4', img_array, fps=2)
