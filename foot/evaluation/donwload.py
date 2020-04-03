import pandas as pd
from PIL import Image
data = pd.read_csv('../newest.csv')

# Download images from url

import urllib.request 
for idx,row_obj in data.iterrows():
    print(row_obj.url)
    if 'png' in row_obj.url:
#         urllib.request.urlretrieve(row_obj.url, str(row_obj.foot_id) + '.png')
        image = Image.open(urllib.request.urlopen(row_obj.url))
        image=image.convert('RGB')
        image.save( 'jpg_images/' + str(row_obj.foot_id) + '.jpg' )
        
    elif 'jpg' in row_obj.url:
#         urllib.request.urlretrieve(row_obj.url, str(row_obj.foot_id) + '.jpg')
        image = Image.open(urllib.request.urlopen(row_obj.url))
        image.save( 'jpg_images/' + str(row_obj.foot_id) + '.jpg' )
        
