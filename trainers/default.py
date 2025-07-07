import os
import numpy as np
import webdataset as wds
import timm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from typing import TypedDict, Optional, Literal
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from helpers.plotters import subplot_metrics, plot_cm
from helpers.checkpoints import save_checkpoint, load_checkpoint, find_last_checkpoint

from models.inception_resnet_v2 import create_inception_resnet_v2
from models.maxvit import create_maxvit_nano_rw_256


class HyperParams(TypedDict):
    lr: float
    weight_decay: float
    step_size: int
    gamma: float


def train(
    epochs,
    data_dir: str,
    dataset_type: Literal["webdataset", "datafolder"] = "datafolder",
    model_name: Literal["inception_resnet_v2",
                        "vgg19", "maxvit_nano_rw_256"] = "inception_resnet_v2",
    hyperparams: HyperParams = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'step_size': 10,
        'gamma': 0.95
    },
    num_classes: int = 3,
    batch_size: int = 64,
    shuffle_size: int = 10000
):
    print(model_name, epochs, dataset_type, data_dir,
          batch_size, hyperparams, "hyperparams")

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.)

    lr: float = hyperparams['lr']
    weight_decay: float = hyperparams['weight_decay']
    step_size: int = hyperparams['step_size']
    gamma: float = hyperparams['gamma']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_creator = {
        'inception_resnet_v2': create_inception_resnet_v2,
        'maxvit_nano_rw_256': create_maxvit_nano_rw_256,
    }

    model = model_creator[model_name](num_classes=num_classes).to(device)

    data_config = timm.data.resolve_model_data_config(model)
    data_transforms = {
        'train': timm.data.create_transform(
            **data_config, is_training=True),
        'val':  timm.data.create_transform(
            **data_config, is_training=False),
    }

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    def get_sample(sample):
        image = data_transforms['train'](sample)
        return image

    def get_sample_val(sample):
        image = data_transforms['val'](sample)
        return image

    def get_label(sample):
        return int(sample) - 5

    image_datasets = {
        'train': wds.WebDataset(os.path.join(data_dir, 'train.tar'))
        .shuffle(shuffle_size)
        .decode("pil")
        .to_tuple("input.jpg", "target.cls")
        .map_tuple(get_sample, get_label),
        'val': wds.WebDataset(os.path.join(data_dir, 'test.tar'))
        .decode("pil")
        .to_tuple("input.jpg", "target.cls")
        .map_tuple(get_sample_val, get_label)
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size)
    }

    dataset_sizes = {
        'train': sum(1 for _ in wds.WebDataset(os.path.join(data_dir, 'train.tar'))),
        'val': sum(1 for _ in wds.WebDataset(os.path.join(data_dir, 'test.tar')))
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_auc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': []
    }

    start_epoch = 0
    best_acc = 0
    best_loss = 0
    best_f1 = 0
    best_auc = 0

    if not os.path.exists(f'{data_dir}/checkpoints'):
        os.makedirs(f'{data_dir}/checkpoints')

    last_epoch = epochs - 1
    checkpoint_path, _ = find_last_checkpoint(data_dir)

    if checkpoint_path:
        print(f"Carregando checkpoint: {checkpoint_path}")
        start_epoch, history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler)
        best_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        best_loss = max(history['val_loss']) if history['val_loss'] else 0.0
        best_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
        best_auc = max(history['val_auc']) if history['val_auc'] else 0.0

        print(f"Retomando treinamento da época {start_epoch + 1}")
    else:
        print("Iniciando novo treinamento")

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_probs = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    probs = torch.nn.functional.softmax(outputs, dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().detach().numpy())
                all_preds.extend(preds.cpu().detach().numpy().flatten())

            epoch_loss = running_loss / dataset_sizes[phase]
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)

            epoch_acc = np.mean(all_preds == all_labels)

            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
            classes = np.unique(
                np.concatenate((all_labels, all_preds)))
            y_bin = label_binarize(all_labels, classes=classes)
            epoch_auc = roc_auc_score(y_bin, all_probs, multi_class='ovr')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                history['train_f1'].append(epoch_f1)
                history['train_auc'].append(epoch_auc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                history['val_f1'].append(epoch_f1)
                history['val_auc'].append(epoch_auc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step()

        is_best = len(history['val_acc']
                      ) > 0 and history['val_acc'][-1] > best_acc

        if (epoch % 10 == 0 or epoch == last_epoch or is_best):
            save_checkpoint(epoch, model, optimizer, scheduler,
                            history, data_dir, is_best=is_best)

        if is_best:
            best_loss = history['val_loss'][-1]
            best_acc = history['val_acc'][-1]
            best_f1 = history['val_f1'][-1]
            best_auc = history['val_auc'][-1]

    best_checkpoint = torch.load(f'{data_dir}/checkpoints/best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Plotar métricas
    subplot_metrics(history, data_dir)

    # Gerar e plotar matriz de confusão no conjunto de validação
    print("Gerando matriz de confusão...")

    plot_cm(model, device, dataloaders['val'], data_dir)

    print("Best: Loss, Acc, F1, AUC")
    print(best_loss, best_acc, best_f1, best_auc)

    return model
