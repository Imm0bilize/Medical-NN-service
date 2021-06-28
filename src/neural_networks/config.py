import os


IMG_SIZE = 512
N_CLASSES = 1
MIN_BOUND = -1000.0
MAX_BOUND = 500.0
VALUE_FOR_SEGMENTED_DMG = 1000
VALUE_FOR_EQUAL_HU = 32768
DICOM_BACKGROUND = -2048
MASK_MERGE_THRESHOLD = 0.1

PATH_TO_LUNG_MODEL_WEIGHTS = os.path.join(os.path.dirname('config.py'),
                                          'models', 'weights', 'lung_segmentation_weights.h5')
PATH_TO_DAMAGE_MODEL_WEIGHTS = os.path.join(os.path.dirname('config.py'),
                                            'models', 'weights', 'damage_segment_weights.h5')