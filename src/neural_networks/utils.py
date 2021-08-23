import cv2
import tensorflow as tf
import numpy as np


def get_all_sys_info(logger):
    logger.info(f'{"#" * 14 } Sys info {"#" * 14}')
    for k, v in tf.sysconfig.get_build_info().items():
        message = f"{k}\t->\t{v}"
        if v:
            logger.info(message)
        else:
            logger.warning(message)
    logger.info(f'{"#" * 10 } Supportable GPU {"#" * 10}')
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in tf.config.list_physical_devices("GPU"):
            logger.info(f"{gpu}")
    else:
        logger.warning("GPU not found")


def resize_predictions(predictions: np.ndarray, original_size):
    return np.array(
        [cv2.resize(prediction, dsize=(original_size, original_size)) for prediction in predictions],
        dtype=np.uint8
    )
