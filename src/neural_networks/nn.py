from abc import ABC, abstractmethod
from typing import List

import cv2
import tensorflow as tf
import numpy as np
import pydicom as dicom
from pydicom.filebase import DicomBytesIO
from PIL import Image

from .models import DamageSegmentation, Yolo
from .error_handler import incorrect_start_sess_param, models_weights_isnt_defined
from .config import MIN_BOUND, MAX_BOUND, DICOM_BACKGROUND, MASK_MERGE_THRESHOLD, IMG_SIZE
from .utils import resize_predictions


class Model(ABC):
    @abstractmethod
    def load_model(self) -> tf.keras.models.Model:
        pass

    @abstractmethod
    def processing_predictions(self, inputs_data: np.ndarray, predictions: np.ndarray, original_size: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_name_working_gpu(self) -> str:
        pass


class YoloStrategy(Model):
    def _bboxes_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def _nms(self, bboxes, iou_threshold=0.45, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest score according to socre order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
                iou = self._bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    def _postprocess_boxes(self, pred_bbox, original_image, score_threshold=0.3, input_size=512):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def _draw_bbox(self, image, bboxes, rectangle_colors=(255, 0, 0)):
        image = (image - np.min(image)) / np.max(image)
        image = image * 255
        image_h, image_w, _ = image.shape

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            bbox_color = rectangle_colors
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)
        return image

    def load_model(self) -> tf.keras.models.Model:
        return Yolo().build_model()

    def get_name_working_gpu(self) -> str:
        return "/GPU:1"

    def processing_predictions(self, inputs_data: np.ndarray, predictions: np.ndarray, original_size: int) -> np.ndarray:
        # TODO ???????????????????? ????????????????
        processed_prediction = []
        n_pred = []
        img_dict = {}
        for arr in predictions:
            for i, sub in enumerate(arr):
                if not i in img_dict:
                    img_dict[i] = []
                sub_arr = img_dict[i]
                sub_arr.append(sub)
        for v in img_dict.values():
            n_pred.append(np.array(v, dtype=object))
        for i, prediction in enumerate(n_pred):
            prediction = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in prediction]
            prediction = tf.concat(prediction, axis=0)

            bboxes = self._postprocess_boxes(prediction, inputs_data[i])
            bboxes = self._nms(bboxes, method='nms')

            processed_prediction.append(self._draw_bbox(inputs_data[i], bboxes))
        if original_size != IMG_SIZE:
            return resize_predictions(processed_prediction, original_size=original_size)
        return np.array(processed_prediction, dtype=np.uint8)


class DamageSegmentationStrategy(Model):
    def _merge_mask_with_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        dcm = (image - np.min(image)) / np.max(image)
        dcm = dcm * 255
        img = Image.fromarray(dcm.astype('uint8'))
        r, g, b = img.split()
        tmp = np.where(np.squeeze(mask) > MASK_MERGE_THRESHOLD, 255, np.array(r))
        r = Image.fromarray(tmp)
        return np.array(Image.merge("RGB", (r, g, b)))

    def processing_predictions(self, inputs_data: np.ndarray, predictions: np.ndarray, original_size: int) -> np.ndarray:
        processed_predictions = np.array(
            [self._merge_mask_with_image(x, y) for x, y in zip(inputs_data, predictions)],
            dtype=np.uint8
        )
        
        if original_size != IMG_SIZE:
            return resize_predictions(processed_predictions, original_size=original_size)
        return processed_predictions

    def load_model(self) -> tf.keras.models.Model:
        return DamageSegmentation().build_model()

    def get_name_working_gpu(self) -> str:
        return "/GPU:0"


class NeuralNetwork:
    def __init__(self, param):
        params = {
            'ct-dmg-seg': DamageSegmentationStrategy,
            'ct-dmg-det': YoloStrategy
        }
        self._strategy = params.get(param,
                                   incorrect_start_sess_param)()
        self._gpu_name = self._strategy.get_name_working_gpu()
        self._image_size = 512

        try:
            self._model = self._strategy.load_model()
        except OSError as e:
            models_weights_isnt_defined()

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

    def _prepare_data(self, path_to_file):
        dcm = dicom.read_file(path_to_file)
        dcm = self._transform_to_hu(dcm)
        dcm = self._normalize(dcm)
        dcm = dcm.astype(np.float32)
        dcm = tf.convert_to_tensor(dcm, dtype=tf.float32)

        if IMG_SIZE != dcm.shape[0]:
            self._image_size = dcm.shape[0]
            dcm = tf.image.resize(dcm, size=[IMG_SIZE, IMG_SIZE])
        dcm = tf.image.grayscale_to_rgb(dcm)
        return dcm

    @tf.autograph.experimental.do_not_convert
    def create_predictions(self, paths_to_data: List[str]) -> np.ndarray:
        data = np.zeros((len(paths_to_data), 512, 512, 3))
        for i,path in enumerate(paths_to_data):
            data[i] = self._prepare_data(path)
        with tf.device(self._gpu_name):
            predictions = self._model.predict(data)
        predictions = self._strategy.processing_predictions(data, predictions, self._image_size)
        return predictions
