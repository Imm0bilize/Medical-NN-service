import atexit
import argparse
from threading import Timer

import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger

import utils
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
        prediction = NN.create_prediction(data)
        logger.debug('Prediction created')
        return utils.get_post_processed_data(data, prediction)
    else:
        logger.error('The session was not created, but there was an attempt to upload file')
        return {'error': 'the session was not created'}


@app.get('/start/{params}')
def start_session(params: str):
    """
    Starting session and timer for auto close session, loading models

    :param params:
     ct-dmg-seg - Segmentation damage on CT,
     ct-dmg-det - Detection damage on CT
    """

    global NN

    try:
        if NN is None:
            NN = NeuralNetwork(params)
        else:
            NN = None
            NN = NeuralNetwork(params)
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
    interesting_pixels = NN.get_num_interesting_pixels()
    pred = None
    logger.debug('Session stopping')
    return {'info': 'Session stopping', 'interesting_pixels': interesting_pixels}


@atexit.register
def stopping_service():
    logger.debug('Stopping service')


@app.get('/')
def start_page():
    return {'info': 'Service working'}


@logger.catch
def main():
    parser = argparse.ArgumentParser(
        description="Service for covid-19 segmentation and detection in dicom files",
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default='localhost')
    parser.add_argument("--port", default=8080)
    args = parser.parse_args()

    logger.debug('Starting service')
    uvicorn.run("app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == '__main__':
    main()
