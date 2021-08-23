import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # warning or error
import pickle
import socket
import atexit
import threading
import time
from datetime import datetime

from loguru import logger

import src.service.utils as utils
from src.neural_networks.nn import NeuralNetwork
from src.neural_networks.utils import get_all_sys_info


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 8005))
server_socket.listen()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.settimeout(60)

while True:
    try:
        client_socket.connect(('mmsys', 8010))
        break
    except ConnectionRefusedError:
        time.sleep(0.5)

client_socket.settimeout(None)

def close_all_connections():
    server_socket.close()
    client_socket.close()


atexit.register(close_all_connections)


logger.add(os.path.join(*os.path.realpath(__file__).split(os.sep)[:-2], 'log', 'debug.log'),
           format='{time} {level} {message}', level='DEBUG',
           rotation='512 KB', compression='zip')


get_all_sys_info(logger)

logger.info(f"{'#' * 14} Build model {'#' * 14}")
try:
    NN_INSTANCE_SEGMENTATION = NeuralNetwork(param="ct-dmg-seg")
    logger.info("Segmentation model loaded")
except ValueError:
    logger.critical("Weights for segmentation model not found")

try:
    NN_INSTANCE_DETECTION = NeuralNetwork(param="ct-dmg-det")
    logger.info("Detection model loaded")
except ValueError:
    logger.critical("Weights for detection model not found")


logger.info(
    f"Service started | Log file: {os.path.join(*os.path.realpath(__file__).split(os.sep)[:-2], 'log', 'debug.log')}"
)


def send_response(research_name, nn_name):
    message = f'{research_name}@{nn_name}'
    client_socket.send(pickle.dumps(message))


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
                # loaded_file = load(path)
                data.append(path)
            except OSError as e:
                logger.error(f'Error: {e} on path {path}, skipped')
        if len(data) != 0:
            predictions = nn_instance.create_predictions(data)
            utils.save_prediction(logger, research_paths[batch_size*idx:batch_size*(idx+1)], predictions,
                                  additions_name=nn_name)
    logger.info(f'Finished {nn_name} predictions for {research_name} | Elapsed time:{datetime.now()-start_time}')
    send_response(research_name, nn_name)


def start_predictions_threads(research_name, research_paths):
    segmentation_thread = threading.Thread(target=create_prediction, args=(NN_INSTANCE_SEGMENTATION, "seg",
                                                                           research_name, research_paths))
    detection_thread = threading.Thread(target=create_prediction, args=(NN_INSTANCE_DETECTION, "det",
                                                                        research_name, research_paths))

    segmentation_thread.start()
    detection_thread.start()
    logger.info(f'Research for {research_name} has been launched')


def prepare_request(request):
    path_to_file_with_paths = request.decode("ASCII")
    with open(path_to_file_with_paths, "rb") as f:
        request = pickle.load(f)
    research_name, paths = request['name'], request['paths']
    return research_name, paths


@logger.catch
def event_loop():
    try:
        while True:
            client, _ = server_socket.accept()
            while True:
                request = client.recv(4096)
                if request:
                    try:
                        research_name, research_paths = prepare_request(request)
                    except KeyError:
                        logger.error("Request processing error")
                        continue
                    start_predictions_threads(research_name, research_paths)
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupted")


if __name__ == '__main__':
    event_loop()
