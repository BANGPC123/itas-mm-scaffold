import cv2
import numpy as np


def insufficient_information(image):
    if image is None or image.size==0:
        return True
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Conservative: reject near-uniform input, not real low-light scenes.
    return int(gray.max())-int(gray.min())<=2


def crop_quality(crop):
    if crop.size==0:return 0.
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    detail=cv2.Laplacian(cv2.resize(gray,(64,64)),cv2.CV_32F).var()
    return float(np.log1p(detail)*min(crop.shape[:2]))
