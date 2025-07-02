import os
import concurrent.futures
import PIL
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np


def detect_faces(input_path, output_path, color_threshold=75):
    mtcnn = MTCNN(keep_all=True, device='cuda:0')

    if not os.path.exists(f'{output_path}'):
        os.makedirs(f'{output_path}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for entry in os.scandir(input_path):
            _, pure_img_path = entry.path.rsplit('/', 1)

            # executor.submit(process_face if is_colorful else detect_face, pure_img_path,
            #                 input_path, output_path, mtcnn, color_threshold)
            executor.submit(detect_face_colorful, pure_img_path,
                            input_path, output_path, mtcnn, color_threshold)

        executor.shutdown(wait=True)


def get_roi(padding, width, height, center_x, center_y):
    padding_x = padding * width
    padding_y = padding * height

    x1 = max(center_x - padding_x, 0)
    y1 = max(center_y - padding_y, 0)
    x2 = min(center_x + padding_x, width)
    y2 = min(center_y + padding_y, height)

    return [x1, y1, x2, y2]


def detect_face_colorful(img_path, input_path, output_path: str, mtcnn: MTCNN, color_threshold=75):
    extension_less_img_path, _ = img_path.rsplit('.', 1)

    full_img_path = f'{input_path}/{img_path}'
    img = Image.open(full_img_path).convert('HSV')
    h, s, v = img.split()

    colored = np.mean(
        s) >= color_threshold if color_threshold is not None else True

    if colored:

        original_image: PIL.Image = Image.open(full_img_path).convert('RGB')
        boxes, probs, points = mtcnn.detect(original_image, landmarks=True)

        if points is not None and probs[0] > .98:
            landmarks = points[0]
            left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
            nose_x, nose_y = nose

            width, height = original_image.size
            roi = get_roi(.4, width, height, nose_x, nose_y)
            new_image = original_image.crop(roi)
            new_image = new_image.resize((300, 300), Image.Resampling.BICUBIC)
            new_image.save(
                f'{output_path}/{extension_less_img_path}.png', 'PNG')
