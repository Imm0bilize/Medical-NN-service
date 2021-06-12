import atexit

import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger


app = FastAPI()
logger.add('log/debug.log', format='{time} {level} {message}',
           level='DEBUG', rotation='00:00', compression='zip')


@app.post('/upload/ct/seg/lung_and_dmg')
def create_lung_and_damage_seg(file: UploadFile = File(...)):
    data = file.read()


@app.post('/upload/ct/seg/lung')
def create_lung_seg(file: UploadFile = File(...)):
    data = file.read()


@app.post('/upload/ct/seg/dmg')
def create_lung_seg(file: UploadFile = File(...)):
    data = file.read()


@app.post('/upload/ct/det/dmg')
def create_ct_damage_detection(file: UploadFile = File(...)):
    data = file.read()


@app.get('/')
def start_page():
    return {'info': 'Service working'}


@atexit.register
def stopping_service():
    logger.debug('Stopping service')


@logger.catch
def main():
    logger.debug('Starting service')
    uvicorn.run("app:app", host="localhost", port=8080, log_level="debug")


if __name__ == '__main__':
    main()
