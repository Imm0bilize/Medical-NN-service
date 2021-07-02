import json
from datetime import datetime
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


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


def convert_to_json(meta: Dict[str, str], image: np.ndarray) -> str:
    data = {'meta': meta, 'image': image.tolist()}
    return json.dumps(data)


def get_post_processed_data(dcm: bytes, mask: np.ndarray) -> str:
    meta = get_dicom_meta(dcm)
    return convert_to_json(meta, mask)
