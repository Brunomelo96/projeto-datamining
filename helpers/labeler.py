import os
import torch
import stone


def get_new_labels(input_path):
    """
    Anota as imagens de acordo com o tom de pele baseado na escala Monk.
    """
    image_paths = []
    preds = []
    labels = []

    with torch.inference_mode() and torch.no_grad():
        for entry in os.scandir(input_path):
            image_path = entry.path.split('/')[-1]
            image_name, _ = entry.path.rsplit('.', 1)
            _, label = image_name.rsplit('_', 1)

            print(image_name, "skin")
            skin_pred = stone.process(entry.path, 'color', 'monk')
            preds.append(skin_pred)
            labels.append(int(label))
            image_paths.append(image_path)

    return preds, labels, image_paths


def relabel_images(input_path, output_path):
    if not os.path.exists(f'{output_path}'):
        os.makedirs(f'{output_path}')

    label_map = {
        '#f6ede4': 1,
        '#f3e7db': 2,
        '#f7ead0': 3,
        '#eadaba': 4,
        '#d7bd96': 5,
        '#a07e56': 6,
        '#825c43': 7,
        '#604134': 8,
        '#3a312a': 9,
        '#292420': 10,
    }

    preds, labels, image_paths = get_new_labels(input_path)

    for i, image_path in enumerate(image_paths):
        image_name, extension = image_path.rsplit('.', 1)
        labeless, label = image_name.rsplit('_', 1)
        face_pred = preds[i]['faces'][0] if len(
            preds[i]['faces']) > 0 else None
        face_id = face_pred['face_id'] if face_pred is not None else None

        if face_id is not None:
            new_label = label_map[str(face_pred['skin_tone']).lower()]
            new_image_path = f'{labeless}_{new_label}.{extension}'
            print(new_image_path, "new_image_path")
            os.system(
                f'cp {input_path}/{image_path} {output_path}/{new_image_path}')
