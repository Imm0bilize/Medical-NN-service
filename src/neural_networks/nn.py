import tensorflow as tf
import tensorflow_io as tfio


import utils
from models.lung_segmentation import LungSegmentation
from models.dmg_segmentation import DamageSegmentation
from config import PATH_TO_DAMAGE_MODEL_WEIGHTS, PATH_TO_LUNG_MODEL_WEIGHTS,\
                   IMG_SIZE, MIN_BOUND, MAX_BOUND, DICOM_BACKGROUND, VALUE_FOR_EQUAL_HU


class NeuralNetwork:
    def __init__(self, param, logger):
        params = {
            'ct-lung-seg': self._load_lung_seg_model,
            'ct-dmg-seg': self._load_dmg_seg_model,
            'ct-lung-dmg-seg': self._create_lung_and_dmg_model,
            'ct-dmg-det': self._load_ct_detection_model
        }
        model_loader = params.get(param, utils.incorrect_start_sess_param)
        self._model = model_loader()
        self._logger = logger

    def __del__(self):
        try:
            del self._model
        except NameError:  # error in start session params
            pass

    def _create_model_with_preprocessing(self, preprocess_func, model):
        inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1))
        preprocessed_input = tf.keras.layers.Lambda(preprocess_func)(inputs)
        predictions = model(preprocessed_input)
        model = tf.keras.models.Model(inputs=[inputs], outputs=[predictions])
        return model

    def _create_lung_and_dmg_model(self):
        dmg_model = self._load_dmg_seg_model()
        lung_model = self._load_lung_seg_model()

        inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1))

        dmg_mask = dmg_model(inputs)
        lung_mask = lung_model(inputs)

        return tf.keras.models.Model(inputs=[inputs], outputs=[lung_mask, dmg_mask])

    def _preprocessing_for_lung_seg(self, inputs):
        data = inputs - DICOM_BACKGROUND
        data = data / 2 ** 16
        return data

    def _preprocessing_dicom(self, inputs):
        data = tf.where(inputs == DICOM_BACKGROUND, 0.0, inputs)
        data = (data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        data = tf.where(data > 1.0, 1.0, data)
        data = tf.where(data < 0.0, 0.0, data)
        data = tf.image.grayscale_to_rgb(data)
        return data

    def _prepare_data(self, data):
        dcm = tfio.image.decode_dicom_image(data)[0]  # remove first dim
        dcm = tf.cast(dcm, dtype=tf.float32)
        dcm = dcm - VALUE_FOR_EQUAL_HU
        dcm = tf.expand_dims(dcm, axis=0)  # add batch dim
        return dcm

    def _load_dmg_seg_model(self):
        dmg_model = DamageSegmentation(PATH_TO_DAMAGE_MODEL_WEIGHTS).build_densenet121_unet()
        return self._create_model_with_preprocessing(preprocess_func=self._preprocessing_dicom,
                                                     model=dmg_model)

    def _load_lung_seg_model(self):
        lung_model = LungSegmentation(PATH_TO_LUNG_MODEL_WEIGHTS).build_unet()
        return self._create_model_with_preprocessing(preprocess_func=self._preprocessing_for_lung_seg,
                                                     model=lung_model)

    def _load_ct_detection_model(self):
        pass

    def create_prediction(self, data):
        data = self._prepare_data(data)
        predictions = self._model.predict(data)
        return predictions
