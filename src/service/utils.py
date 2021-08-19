import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pydicom as dicom
from pydicom.filebase import DicomBytesIO
from PIL import Image


def get_dicom_meta(dcm: bytes) -> Dict[str, str]:
    dcm = dicom.dcmread(DicomBytesIO(dcm))

    meta = {}
    tags = ('InstitutionAddress', 'PatientName', 'PatientBirthDate', 'StudyDate', 'StudyInstanceUID')

    for tag in tags:
        if tag in ('PatientBirthDate', 'StudyDate'):
            tmp = dcm.get(tag)
            meta.update({tag: f"{tmp[6:]}.{tmp[4:6]}.{tmp[:4]}"})  # to day.month.year
        else:
            meta.update({tag: str(dcm.get(tag))})

    meta.update({'AnalysisDate': datetime.now().strftime('%d.%m.%Y %H:%M')})
    return meta


def convert_to_json(meta: Dict[str, str], image: np.ndarray) -> str:
    data = {'meta': meta, 'image': image.tolist()}
    return json.dumps(data)


def get_post_processed_data(dcms: List[bytes], masks: np.ndarray) -> str:
    meta = get_dicom_meta(dcms[0])  # get meta from one file
    return convert_to_json(meta, masks)


def save_prediction(logger, paths, predictions, additions_name):
    for path, prediction in zip(paths, predictions):
        try:
            img = Image.fromarray(prediction)
            img.save(path[:-4] + '-' + additions_name + '.png')
        except OSError:
            logger.error(f'File on path {path} don`t saved, skipped...')
