import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from src.neural_networks.config import IMG_SIZE


def get_dicom_meta(dcm: bytes) -> Dict[str, str]:
    tags = tf.constant([
        tfio.image.dicom_tags.InstitutionAddress,
        tfio.image.dicom_tags.PatientsName,
        tfio.image.dicom_tags.PatientsBirthDate,
        tfio.image.dicom_tags.StudyDate,
        tfio.image.dicom_tags.StudyInstanceUID,
    ], dtype=tf.uint32)
    file_meta = tfio.image.decode_dicom_data(dcm, tags=tags).numpy()

    tags_name = ['InstitutionAddress', 'PatientsName', 'PatientsBirthDate', 'StudyDate', 'StudyInstanceUID']
    meta = {}

    for tag_name, value in zip(tags_name, file_meta):
        if tag_name in ('PatientsBirthDate', 'StudyDate'):
            tmp = value.decode('UTF-8')
            tmp = f"{tmp[6:]}.{tmp[4:6]}.{tmp[:4]}"  # day month  year
            meta.update({tag_name: tmp})
        else:
            meta.update({tag_name: value.decode('UTF-8')})

    meta.update({'AnalysisDate': datetime.now().strftime('%d.%m.%Y %H:%M')})
    return meta


def merge_mask_with_image(image: bytes, mask: np.ndarray,
                          segment_rgb_color: List[float] = [1.0, 0.0, 0.0]) -> np.ndarray:

    image = tfio.image.decode_dicom_image(image)[0]  # for shape (SIZE, SIZE, 1)
    image = tf.cast(image, dtype=tf.float32)
    image = image - 30720  # TFIO dicom shift
    image = image / np.max(image)
    image = tf.image.grayscale_to_rgb(image)

    tmp_red_image = tf.convert_to_tensor(
                    np.array([[segment_rgb_color for _ in range(IMG_SIZE)] for _ in range(IMG_SIZE)]),
                    dtype=tf.float32
    )

    merged_image = tf.where(mask > 0.0, tmp_red_image, image)
    return merged_image.numpy()


def convert_to_json(meta: Dict[str, str], image: tf.Tensor) -> str:
    data = {'meta': meta, 'image': image.tobytes()}
    return json.dump(data)


def get_post_processed_data(dcm: bytes, mask: np.ndarray) -> str:
    image = merge_mask_with_image(dcm, mask)
    meta = get_dicom_meta(dcm)
    return convert_to_json(meta, image)