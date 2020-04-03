import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import pandas as pd
import cv2
import imutils
from PIL import Image

data = pd.read_csv('../newest.csv')

# Combining all above for all images to make csv for training pipeline
z='.jpg'
right_points_x = [i for i in data.columns if 'r_' in i and 'x' in i and 'v' not in i and 'p' not in i]
right_points_y = [i for i in data.columns if 'r_' in i and 'y' in i and 'v' not in i and 'p' not in i]
left_points_x = [i for i in data.columns if 'l_' in i and 'x' in i and 'v' not in i and 'p' not in i]
left_points_y = [i for i in data.columns if 'l_' in i and 'y' in i and 'v' not in i and 'p' not in i]

xml_list_lr = []



for idx in range(data.shape[0]):
#if str(data.iloc[idx].foot_id)+z not in arr:
    #    continue
#     print(idx)
    r_max_x=-99 
    r_min_x=100 
    r_max_y=-99
    r_min_y=100

    l_max_x=-99 
    l_min_x=100 
    l_max_y=-99
    l_min_y=100

    for i in right_points_x:
        r_max_x = max(r_max_x, min(0.48,data.iloc[idx][i]))
        r_min_x = min(r_min_x, max(-0.48,data.iloc[idx][i]))

    for i in right_points_y:
        r_max_y = max(r_max_y, min(0.48,data.iloc[idx][i]))
        r_min_y = min(r_min_y, max(-0.48,data.iloc[idx][i]))

    for i in left_points_x:
        l_max_x = max(l_max_x, min(0.48,data.iloc[idx][i]))
        l_min_x = min(l_min_x, max(-0.48,data.iloc[idx][i]))

    for i in left_points_y:
        l_max_y = max(l_max_y, min(0.48,data.iloc[idx][i]))
        l_min_y = min(l_min_y, max(-0.48,data.iloc[idx][i]))
   
 
    [l_max_x, l_min_x, l_max_y, l_min_y] = np.multiply([l_max_x, l_min_x, l_max_y, l_min_y],512)
    [r_max_x, r_min_x, r_max_y, r_min_y] = np.multiply([r_max_x, r_min_x, r_max_y, r_min_y],512)
   

    r_min_x = int(r_min_x)+256
    r_min_y = 256-int(r_min_y)
    r_max_x = int(r_max_x)+256
    r_max_y = 256-int(r_max_y)

    r_max_y, r_min_y = r_min_y, r_max_y
   
    l_min_x = int(l_min_x)+256
    l_min_y = 256-int(l_min_y)
    l_max_x = int(l_max_x)+256
    l_max_y = 256-int(l_max_y)

    l_max_y, l_min_y = l_min_y, l_max_y
   
    if data.iloc[idx].l_probability==0:
        l_min_x,l_min_y,l_max_x,l_max_y=0,0,0,0
    if data.iloc[idx].r_probability==0:
        r_min_x,r_min_y,r_max_x,r_max_y=0,0,0,0
    
    print('left---->',l_min_x,l_min_y,l_max_x,l_max_y,end="     ")
    print('right---->',r_min_x,r_min_y,r_max_x,r_max_y)
    xml_list_lr.append([str(data.iloc[idx].foot_id)+z,520,520,'left_foot',l_min_x,l_min_y,l_max_x,l_max_y,'right_foot',r_min_x,r_min_y,r_max_x,r_max_y])


column_name = ['filename', 'width', 'height',
                'left_foot', 'lxmin', 'lymin', 'lxmax', 'lymax','right_foot', 'rxmin', 'rymin', 'rxmax', 'rymax',]

lr_df = pd.DataFrame(xml_list_lr, columns=column_name)
lr_df.to_csv('lr_foot.csv', index=None)
