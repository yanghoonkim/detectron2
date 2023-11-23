import numpy as np
import cv2
import pycocotools.mask as mask_util

def build_mask_from_seg_pieces(pieces):
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    for seg_piece in pieces:
        points = np.array([(p['x'], p['y']) for p in seg_piece], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask

def build_segmentation_from_multipoly(multipoly):
    points = multipoly['coord']['points']
    masks = [build_mask_from_seg_pieces(p) for p in points]
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    segmentation = mask_util.encode(np.array(combined_mask, order="F", dtype=np.uint8))
    segmentation['counts'] = segmentation['counts'].decode('utf8')
    
    return segmentation