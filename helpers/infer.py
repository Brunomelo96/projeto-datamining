import os
import torch
from helpers.checkpoints import load_checkpoint
from models.maxvit import create_maxvit_nano_rw_256


def infer_model(
    checkpoint_base_path: str,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_maxvit_nano_rw_256(num_classes=4).to(device)

    checkpoint_path = os.path.join(
        checkpoint_base_path, 'best_model.pth')

    start_epoch, history = load_checkpoint(
        checkpoint_path, model)

    print('Best epoch', start_epoch)

    print('TRAIN')
    print('ACC, LOSS, F1, AUC')
    print(
        f'({round(history["train_acc"][-1], 4)}, {round(history["train_loss"][-1], 4)}, {round(history["train_f1"][-1], 4)}, {round(history["train_auc"][-1], 4)})')

    print('VAL')
    print('ACC, LOSS, F1, AUC')
    print(
        f'({round(history["val_acc"][-1], 4)}, {round(history["val_loss"][-1], 4)}, {round(history["val_f1"][-1], 4)}, {round(history["val_auc"][-1], 4)})')
