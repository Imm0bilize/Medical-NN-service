import tensorflow as tf


def get_all_sys_info(logger):
    logger.debug(f'{"#" * 14 } Sys info {"#" * 14}')
    for k, v in tf.sysconfig.get_build_info().items():
        message = f"{k}\t->\t{v}"
        if v:
            logger.debug(message)
        else:
            logger.warning(message)
    logger.debug(f'{"#" * 10 } Supportable GPU {"#" * 10}')
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in tf.config.list_physical_devices("GPU"):
            logger.debug(f"{gpu}")
    else:
        logger.warning("GPU not found")
