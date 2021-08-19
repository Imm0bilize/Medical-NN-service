import os
import threading
import socket
from typing import List
from datetime import datetime

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

import src.service.utils as utils
from src.neural_networks.nn import NeuralNetwork
from src.neural_networks.utils import get_all_sys_info

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # warning or error


app = FastAPI(on_shutdown=[lambda: logger.debug('Service stopped')])


logger.add(os.path.join(*os.path.realpath(__file__).split(os.sep)[:-2], 'log', 'debug.log'),
           format='{time} {level} {message}', level='DEBUG',
           rotation='512 KB', compression='zip')


get_all_sys_info(logger)

logger.debug(f"{'#' * 14} Build model {'#' * 14}")
try:
    NN_INSTANCE_SEGMENTATION = NeuralNetwork(param="ct-dmg-seg")
    logger.debug("Segmentation model loaded")
except ValueError:
    logger.critical("Weights for segmentation model not found")

try:
    NN_INSTANCE_DETECTION = NeuralNetwork(param="ct-dmg-det")
    logger.debug("Detection model loaded")
except ValueError:
    logger.critical("Weights for detection model not found")
logger.debug("#" * 41)

logger.debug(
    f"Service started | Log file: {os.path.join(*os.path.realpath(__file__).split(os.sep)[:-2], 'log', 'debug.log')}"
)


class Item(BaseModel):
    name: str
    paths: List[str]


def post_message_on_socket(research_name, nn_name):
    sock = socket.socket()
    try:
        sock.connect(('localhost', 9090))
        sock.send(f'{research_name}@{nn_name}'.encode())
    except ConnectionRefusedError:
        print(1)
    finally:
        socket.close()


def load(path):
    with open(path, 'rb') as file:
        return file.read()


@logger.catch
def create_prediction(nn_instance, nn_name, research_name, research_paths):
    batch_size = 4
    start_time = datetime.now()
    for idx in range((len(research_paths)//batch_size)+1):
        data = []
        for path in research_paths[batch_size*idx:batch_size*(idx+1)]:
            try:
                loaded_file = load(path)
                data.append(loaded_file)
            except OSError as e:
                logger.error(f'Error: {e} on path {path}, skipped')
        if len(data) != 0:
            predictions = nn_instance.create_predictions(data)
            utils.save_prediction(logger, research_paths[batch_size*idx:batch_size*(idx+1)], predictions,
                                  additions_name=nn_name)
    logger.debug(f'Finished {nn_name} predictions for {research_name} | Elapsed time:{datetime.now()-start_time}')



@app.post('/upload')
def upload_file(item: Item):
    segmentation_thread = threading.Thread(target=create_prediction, args=(NN_INSTANCE_SEGMENTATION, "dmg-seg",
                                                                           item.name, item.paths))
    detection_thread = threading.Thread(target=create_prediction, args=(NN_INSTANCE_DETECTION, "dmg-det",
                                                                        item.name, item.paths))

    segmentation_thread.start()
    detection_thread.start()
    message = f'Research for {item.name} has been launched'
    logger.debug(message)
    return {'info': message}


@app.get('/')
def start_page():
    return {'info': 'Service working'}

