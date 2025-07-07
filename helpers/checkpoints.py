from typing import Dict
import torch
import glob


def save_model(filepath, epoch, loss, model, optimizer, model_dict: Dict):
    data = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': model_dict['train_loss'],
        'test_loss': model_dict['test_loss'],
        'train_acc': model_dict['train_acc'],
        'test_acc': model_dict['test_acc'],
        'precision': model_dict['precision'],
        'recall': model_dict['recall'],
        'f1': model_dict['f1']
    }

    torch.save(data, filepath)


def save_checkpoint(epoch, model, optimizer, scheduler, history, data_dir, is_best=False):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history
    }
    torch.save(state, f'{data_dir}/checkpoints/checkpoint_epoch_{epoch}.pth')
    if is_best:
        torch.save(state, f'{data_dir}/checkpoints/best_model.pth')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['history']


def find_last_checkpoint(data_dir):
    checkpoint_files = glob.glob(
        f'{data_dir}/checkpoints/checkpoint_epoch_*.pth')
    if not checkpoint_files:
        return None, 0

    # Extrair números de época dos nomes de arquivo
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    last_epoch = max(epochs)
    return f'{data_dir}/checkpoints/checkpoint_epoch_{last_epoch}.pth', last_epoch
