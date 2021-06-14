import atexit

import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from threading import Timer

from src.neural_networks.prediction import Prediction


TIME_TO_SHUTDOWN_SESSION: float = 300.0

timer = None
pred = None
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
    data = await file.read()
    # prediction = await model.predict(data)
    set_timer()


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

    global pred

    try:
        pred = Prediction(params, logger)
    except KeyError:
        logger.error(f'Can`t start session with this params -- {params}')
        pred = None
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
