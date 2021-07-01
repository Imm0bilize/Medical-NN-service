import requests


with open('/Users/nikolay/PythonProjects/lung_segmentation_final/src/dataset/val/17/original/IM00200.dcm', 'rb') as file:
    data = file.read()



req = requests.get('http://localhost:8080/')
print(req.text)

req = requests.get('http://localhost:8080/start/ct-dmg-seg')
print(req.text)

req = requests.post('http://localhost:8080/upload', files={'file': data})
pred = req.json()

with open('pred.json', 'w') as file:
    file.write(pred)

req = requests.get('http://localhost:8080/close')
print(req.text)