import atexit
from threading import Timer

import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger

from utils import get_post_processed_data
from src.neural_networks.nn import NeuralNetwork


# in second, default 5 min
TIME_TO_SHUTDOWN_SESSION: float = 300.0

timer = None
NN = None
app = FastAPI()
logger.add('log/debug.log', format='{time} {level} {message}',
           level='DEBUG', rotation='00:00', compression='zip')


def set_timer():
    global timer
    if timer is not None:
        timer.cancel()
    timer = Timer(TIME_TO_SHUTDOWN_SESSION, close_session)
    timer.start()


@app.post('/upload')
async def create_prediction(file: UploadFile = File(...)):
    if NN is not None:
        set_timer()
        data = await file.read()
        mask = NN.create_prediction(data)
        return get_post_processed_data(data, mask)
    else:
        logger.warning('The session was not created, but there was an attempt to upload file')
        return {'error': 'the session was not created'}


@app.get('/start/{params}')
def start_session(params: str):
    """
    Starting session and timer for auto close session, loading models

    :param params:
     ct-lung-seg - Segmentation lung on CT,
     ct-dmg-seg - Segmentation damage on CT,
     ct-lung-dmg-seg - Segmentation lung and damage on CT and calculate damage percent
     ct-dmg-det - Detection damage on CT
    """

    global NN

    try:
        NN = NeuralNetwork(params, logger)
    except KeyError:
        logger.error(f'Can`t start session with this params -- {params}')
        NN = None
        return {'error': 'Starting session interrupted'}

    set_timer()
    logger.debug(f'Starting session: {params}')
    return {'info': 'Session starting'}


@app.get('/close')
def close_session():
    global pred
    pred = None
    logger.debug('Session stopping')
    return {'info': 'Session stopping'}


@atexit.register
def stopping_service():
    logger.debug('Stopping service')


@app.get('/')
def start_page():
    return {'info': 'Service working'}


@logger.catch
def main():
    logger.debug('Starting service')
    uvicorn.run("app:app", host="localhost", port=8080, log_level="debug")


if __name__ == '__main__':
    main()
