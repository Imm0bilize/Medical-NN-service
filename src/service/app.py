import os
from threading import Timer
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from pydantic import BaseModel

import src.service.utils as utils
from src.service.argparser import args
from src.neural_networks.nn import NeuralNetwork


# in second, default 5 min
TIME_TO_SHUTDOWN_SESSION: float = float(args.time_to_shutdown_session)

timer = None
NN = None

logger.add(os.path.join(args.path_to_log_dir, 'debug.log'), format='{time} {level} {message}',
           level='DEBUG', rotation='512 KB', compression='zip')


class Item(BaseModel):
    name: str
    paths: List[str]


def stopping_service():
    logger.debug('Stopping service')


app = FastAPI(on_shutdown=[stopping_service])


def set_timer():
    global timer
    if timer is not None:
        timer.cancel()
    timer = Timer(TIME_TO_SHUTDOWN_SESSION, close_session)
    timer.start()


def create_prediction(data):
    if NN is not None:
        set_timer()
        prediction = NN.create_predictions(data)
        logger.debug('Prediction created')
        return prediction
    else:
        raise ValueError


@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    try:
        prediction = create_prediction(data)
        return utils.get_post_processed_data(data, prediction)
    except ValueError:
        logger.error('The session was not created, but there was an attempt to upload file')
        return {'error': 'the session was not created'}


@app.post('/upload_local')
def upload_local_file(item: Item):
    def load(path):
        with open(path, 'rb') as file:
            return file.read()

    batch_size = 16
    paths = item.paths
    strategy_name = NN.get_strategy()
    for idx in range((len(paths)//batch_size)+1):
        data = []
        for path in paths[batch_size*idx:batch_size*(idx+1)]:
            try:
                loaded_file = load(path)
                data.append(loaded_file)
            except OSError as e:
                logger.error(f'Error: {e} on path {path}, skipped')

        if len(data) != 0:
            predictions = create_prediction(data)
            utils.save_prediction(paths[batch_size*idx:batch_size*(idx+1)],
                                  predictions, strategy_name,
                                  folder_name=item.name)
    return {'info': 'Prediction created'}
            

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
        NN = NeuralNetwork(params)
    except KeyError:
        logger.error(f'Can`t start session with this params -- {params}')
        NN = None
        return {'error': 'Starting session interrupted'}
    except ValueError:
        logger.error(f'Model weights for params [{params}] could not be loaded')
        NN = None
        return {'error': 'Starting session interrupted'}

    set_timer()
    logger.debug(f'Starting session: {params}')
    return {'info': 'Session starting'}


@app.get('/close')
def close_session():
    global NN
    NN = None
    logger.debug('Session stopping')
    return {'info': 'Session stopping'}


@app.get('/')
def start_page():
    return {'info': 'Service working'}


@logger.catch
def main():
    logger.debug(f"\nStarting service with params:\n"
                 f"\tHost: {args.host}, Port: 8080\n"
                 f"\tCheck the working capacity : http://{args.host}:8080/\n"
                 f"\tLog file: {os.path.join(args.path_to_log_dir, 'debug.log')}")
    uvicorn.run("src.service.app:app", host=args.host, port=8080, log_level="warning")


if __name__ == '__main__':
    main()
