import os
import random
import numpy as np
import webdataset as wds


def get_label(img_path: str):
    image_name, _ = img_path.rsplit('.', 1)
    _, label = image_name.rsplit('_', 1)
    return int(label)


def group_by_labels(input_path):
    data = {}
    for entry in os.scandir(input_path):
        label = get_label(entry.path)

        if (label in data):
            data[label] = data[label] + 1
        else:
            data[label] = 1

    return data


def get_image_name(img_path: str):
    name, _ = img_path.rsplit('.', 1)
    return name


def write_wds_dataset(input_path: str, train_path: str, test_path: str, test_size=.2, max_count=None, balance=True, labels_to_ignore=None):
    root_path = train_path.rsplit('/', 1)[0]

    if not os.path.exists(f'{root_path}'):
        os.makedirs(f'{root_path}')

    train_sink = wds.TarWriter(train_path, encoder=False,)
    test_sink = wds.TarWriter(test_path, encoder=False,)

    grouped = group_by_labels(input_path)
    print(grouped, "grouped")
    if labels_to_ignore is not None:
        for label in labels_to_ignore:
            if label in grouped:
                del grouped[label]

    balancer = min(grouped.items(), key=lambda x: x[1])[0]

    count = {}
    totals = {}

    for key, value in grouped.items():
        balance_by = grouped[balancer] if balance else None
        values = [max_count, value, balance_by]
        total = np.min(list(filter(lambda x: x is not None, values)))
        count[key] = 0
        totals[key] = total

    train_count = 0
    test_count = 0

    for entry in os.scandir(input_path):
        _, pure_img_path = entry.path.rsplit('/', 1)
        print(pure_img_path, "pure_img_path")
        print(grouped, "grouped")
        print(totals, "totals")
        print(train_count, "train_count")
        print(test_count, "test_count")

        label = get_label(pure_img_path)

        pick = random.random()

        count_values = list(count.values())
        total_values = list(totals.values())

        if labels_to_ignore is not None and int(label) in labels_to_ignore:
            continue

        if count_values == total_values:
            return

        if count[label] == totals[label]:
            continue

        if pick > test_size:
            train_count += 1
            write_image(pure_img_path, input_path, train_sink)
        else:
            test_count += 1
            write_image(pure_img_path, input_path, test_sink)

        count[label] += 1


def write_image(img_path: str, input_path: str, sink: wds.TarWriter):
    full_img_path = f'{input_path}/{img_path}'
    label = get_label(img_path)

    img_name = get_image_name(img_path)
    with open(full_img_path, "rb") as f:
        image_data = f.read()

        sink.write({
            '__key__': img_name,
            'input.jpg': image_data,
            'target.cls': str(label).encode('utf-8'),
        })
