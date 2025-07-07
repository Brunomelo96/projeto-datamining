from functools import reduce
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_hsv_histogram(input_path, output_path):

    plt.title('Saturation')

    total_s = []

    for entry in os.scandir(input_path):
        print(entry.path, "path")
        img = cv2.imread(entry.path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s_hist = cv2.calcHist([s], [0], None, [256], [0, 256])
        total_s.extend(list(reduce(lambda x, y: x + y, s.tolist())))
        plt.plot(s_hist)

    mean = np.round(np.mean(total_s), 3)
    sd = np.round(np.std(total_s), 3)

    plt.figtext(.8, .8,
                f'Mean: {mean}\nSD: {sd}')

    plt.savefig(f'{output_path}/saturation.png')
    plt.close()


def group_by_labels(input_path):
    data = {}
    for entry in os.scandir(input_path):
        _, pure_img_path = entry.path.rsplit('/', 1)
        path_without_extensions, _ = pure_img_path.rsplit('.')
        _, label = path_without_extensions.rsplit('_', 1)

        if (label in data):
            data[label] = data[label] + 1
        else:
            data[label] = 1

    # branco
    # preto
    # pardo
    # amarelo
    # indigena
    return data


def plot_label_histogram(input_path, output_path):
    plt.title('Label')

    total_labels = []

    for entry in os.scandir(input_path):
        entry_path, _ = entry.path.rsplit('.', 1)
        _, label = entry_path.rsplit('_', 1)

        print(entry_path, label)
        total_labels.append(int(label))

    grouped_labels = group_by_labels(input_path)

    plt.figtext(.1, .95,
                f'Dict: {grouped_labels}')

    plt.hist(total_labels, range=(0, 7))
    plt.savefig(f'{output_path}/label.png')
    plt.close()


def plot_pca_3d(output_path, data, labels=None):
    if not os.path.exists(f'{output_path}'):
        os.makedirs(f'{output_path}')

    if data.shape[1] < 3:
        raise ValueError(
            "PCA data must have at least 3 components for 3D plotting")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(unique_labels)))
        print(colors, "colors")

        for i, label in enumerate(unique_labels):
            print(i, "i")
            indices = [j for j, l in enumerate(labels) if l == label]
            ax.scatter(
                data[indices, 0],
                data[indices, 1],
                data[indices, 2],
                c=[colors[i]],
                label=f'Label {label}',
                alpha=0.7
            )
            ax.legend(labels=['Branco', 'Preto', 'Pardo'])

    # ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA 3D')
    plt.savefig(f'{output_path}/pca_3d.png')
    plt.close()


def subplot_metrics(history, data_dir):
    plt.figure(figsize=(15, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1-Score
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train F1-Score')
    plt.plot(history['val_f1'], label='Validation F1-Score')
    plt.title('Training and Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()

    # Plot AUC-ROC
    plt.subplot(2, 2, 4)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('Training and Validation AUC-ROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{data_dir}/checkpoints/training_metrics.png')
    plt.show()


def plot_cm(model, device, dataloader, data_dir):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            # probs = torch.sigmoid(outputs)
            # preds = (probs > 0.5).float().view(-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calcular matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['0', '1', '2', '3'])

    # Plotar e salvar
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(f'{data_dir}/checkpoints/confusion_matrix.png')
    plt.show()

    return cm


def plot_simple_cm(labels, preds, output_path):
    # 01: Branca;
    # 02: Preta;
    # 03: Parda;
    cm = confusion_matrix(labels, preds)
    print(cm, "cm")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['1', '2', '3'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(f'{output_path}/cluster_cm.png')
    plt.close()
