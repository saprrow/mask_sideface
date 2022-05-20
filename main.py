import logging.config
import os.path

logging.config.fileConfig("./config/logging.conf")
logger = logging.getLogger('api')

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from face_det import FaceDet
from face_align import FaceAlign
from face_masker import FaceMasker

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--input-image', type=str)
    parser.add_argument('--template', type=str, default="1.png")
    parser.add_argument('--output-path', type=str, default="result")
    parser.add_argument('--show-result', type=bool, default=False)
    args = parser.parse_args()
    return args

def draw_bboxes(img, bboxes):
    cv2.line(img, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[1]), color=(255,0,0))
    cv2.line(img, (bboxes[0], bboxes[1]), (bboxes[0], bboxes[3]), color=(255,0,0))
    cv2.line(img, (bboxes[2], bboxes[1]), (bboxes[2], bboxes[3]), color=(255,0,0))
    cv2.line(img, (bboxes[0], bboxes[3]), (bboxes[2], bboxes[3]), color=(255,0,0))

def draw_kpt(img, kpts):
    for (x, y) in kpts.astype(np.int32):
         cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

def main():
    args = parse_args()
    show_result = args.show_result
    image_path = args.input_image
    mask_template_name = args.template
    save_path = os.path.join(args.output_path, "masked_"+image_path.split("/")[-1])

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    face_detector = FaceDet()
    face_align = FaceAlign()

    bboxes = face_detector(image)

    image_show = image.copy()
    face_lms = []
    for box in bboxes:
        # print(box)
        landmarks = face_align(image, box)
        #print(landmarks, landmarks.shape)
        draw_kpt(image_show, landmarks)
        lms = np.reshape(landmarks.astype(np.int32), (-1))
        # print(lms, lms.shape)
        face_lms.append(lms)
        draw_bboxes(image_show, box)
        draw_kpt(image_show, landmarks)
        if show_result:
            cv2.imshow('lms', image_show)
            cv2.waitKey(0)

    # face masker
    is_aug = True
    face_masker = FaceMasker(is_aug)
    # ======masked one face========
    new_image = face_masker.add_mask_one(image, face_lms[0], mask_template_name, mask_template_name)
    cv2.imwrite(save_path, new_image)
    if show_result:
        plt.imshow(new_image)
        plt.show()

if __name__ == '__main__':
    main()