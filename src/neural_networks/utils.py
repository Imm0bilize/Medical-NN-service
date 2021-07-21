import tensorflow as tf

from .config import GPU_LIST


def use_multiple_gpu(func):
    def wrapper(*args, **kwargs):
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=GPU_LIST)
        with mirrored_strategy.scope():
            func(*args, **kwargs)
    return wrapper
