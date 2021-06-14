import tensorflow as tf

import models
from config import PATH_TO_DAMAGE_MODEL_WEIGHTS, PATH_TO_LUNG_MODEL_WEIGHTS,\
                   IMG_SIZE, MIN_BOUND, MAX_BOUND


class Prediction:
    def __init__(self,param, logger):
        params = {
            'ct-lung-seg': self._load_lung_seg_model,
            'ct-dmg-seg': self._load_dmg_seg_model,
            'ct-lung-dmg-seg': self._create_lung_and_dmg_model,
            'ct-dmg-det': self._load_ct_detection_model
        }
        model_loader = params.get(param, self.error_handler)
        self.model = model_loader()
        self.logger = logger

    def error_handler(self, param):
        raise KeyError(f'No logic for the key: {param}')

    def _create_lung_and_dmg_model(self):
        dmg_model = self._load_dmg_seg_model()
        lung_model = self._load_lung_seg_model()

        inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1))
        preproc_data_for_lung = tf.keras.layers.Lambda(self._preprocessing_for_lung_seg)(inputs)
        preproc_data_for_dmg = tf.keras.layers.Lambda(self._preprocessing_for_dmg_seg)(inputs)

        dmg_mask = dmg_model(preproc_data_for_dmg)
        lung_mask = lung_model(preproc_data_for_lung)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[lung_mask, dmg_mask])
        return model

    def _preprocessing_for_lung_seg(self, inputs):
        data = inputs + 2048
        data = data / 2 ** 16
        return data

    def _preprocessing_dicom(self, inputs):
        inputs[inputs == -2048] = 0.0
        data = (inputs - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        data[data > 1.0] = 1.0
        data[data < 0.0] = 0.0
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        data = tf.image.grayscale_to_rgb(data)
        return data

    def _load_dmg_seg_model(self):
        return models.DamageSegmentation(PATH_TO_DAMAGE_MODEL_WEIGHTS)

    def _load_lung_seg_model(self):
        return models.LungSegmentation(PATH_TO_LUNG_MODEL_WEIGHTS)

    def _load_ct_detection_model(self):
        pass

    async def predict(self, data):
        pass
