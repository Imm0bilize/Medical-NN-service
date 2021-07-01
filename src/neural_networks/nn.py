from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import pydicom as dicom
from pydicom.filebase import DicomBytesIO
from PIL import Image

from .models import DamageSegmentation, Yolo
from .error_handler import incorrect_start_sess_param
from .config import MIN_BOUND, MAX_BOUND, DICOM_BACKGROUND, MASK_MERGE_THRESHOLD


class IModel(ABC):
    @abstractmethod
    def load_model(self) -> tf.keras.models.Model:
        pass

    @abstractmethod
    def processing_prediction(self, input_data: np.ndarray, prediction: np.ndarray) -> Image:
        pass


class YoloStrategy(IModel):
    def load_model(self) -> tf.keras.models.Model:
        return Yolo().build_model()

    def processing_prediction(self, input_data: np.ndarray, prediction: np.ndarray) -> Image:
        tf.image.draw_bounding_boxes()
        return


class DamageSegmentationStrategy(IModel):
    def _merge_mask_with_image(self, image: np.ndarray, mask: np.ndarray) -> Image:
        dcm = (image - np.min(image)) / np.max(image)
        dcm = dcm * 255
        img = Image.fromarray(dcm.astype('uint8'))
        r, g, b = img.split()
        tmp = np.where(np.squeeze(mask) > MASK_MERGE_THRESHOLD, 255, np.array(r))
        r = Image.fromarray(tmp)
        return Image.merge("RGB", (r, g, b))

    def processing_prediction(self, input_data: np.ndarray, prediction: np.ndarray) -> Image:
        return self._merge_mask_with_image(input_data, prediction)

    def load_model(self) -> tf.keras.models.Model:
        return DamageSegmentation().build_model()


class NeuralNetwork:
    def __init__(self, param):
        params = {
            'ct-dmg-seg': DamageSegmentationStrategy,
            'ct-dmg-det': YoloStrategy
        }
        self.strategy = params.get(param,
                                   incorrect_start_sess_param)()
        self._model = self.strategy.load_model()

    def _set_outside_scanner_to_air(self, raw_pixel):
        raw_pixel[raw_pixel <= -1000] = 0
        return raw_pixel

    def _transform_to_hu(self, slice):
        images = slice.pixel_array
        images = images.astype(np.int16)

        images = self._set_outside_scanner_to_air(images)

        intercept = slice.RescaleIntercept
        slope = slice.RescaleSlope

        if slope != 1:
            images = slope * images.astype(np.float64)
            images = images.astype(np.int16)
        images += np.int16(intercept)
        
        return np.array(images, dtype=np.int16)

    def _normalize(self, data):
        data = np.expand_dims(data, axis=-1)
        data = np.where(data == DICOM_BACKGROUND, 0.0, data)
        data = (data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        data = np.where(data > 1.0, 1.0, data)
        data = np.where(data < 0.0, 0.0, data)
        return data

    def _prepare_data(self, raw_data: bytes):
        raw = DicomBytesIO(raw_data)
        dcm = dicom.dcmread(raw)
        dcm = self._transform_to_hu(dcm)
        dcm = self._normalize(dcm)
        dcm = tf.convert_to_tensor(dcm)
        dcm = tf.image.grayscale_to_rgb(dcm)
        dcm = tf.expand_dims(dcm, axis=0)
        return dcm

    @tf.autograph.experimental.do_not_convert
    def create_prediction(self, raw_data: bytes):
        data = self._prepare_data(raw_data)
        prediction = self._model.predict(data)
        print(type(prediction))
        prediction = self.strategy.processing_prediction(data.numpy()[0],
                                                         prediction[0])
        return prediction
