import requests
from glob import glob
import json
import numpy as np
import tensorflow as tf


PATH_TO_DIR_WITH_DATA = ''

res = requests.get('http://127.0.0.1:8080/')
print(res)

res = requests.get('http://127.0.0.1:8080/start/ct-dmg-det')
print('Start ', res)

for i, path in enumerate(sorted(glob(f'{PATH_TO_DIR_WITH_DATA}/*.dcm'))):
    with open(path, 'rb') as file:
        res = requests.post('http://127.0.0.1:8080/upload', files={'file': file.read()})

        image = json.loads(res.json())['image']
        tf.keras.preprocessing.image.save_img(path=f'/Users/nikolay/Desktop/res_det/{i}.png', x=np.array(image))


res = requests.get('http://127.0.0.1:8080/close')
print('End ', res.json())
