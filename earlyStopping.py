import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_name='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
                            在最后一次验证集上loss开始增加后等待多久停止
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
                            提示信息，打印信息
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
                            监控指标上的改变幅度
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_name = checkpoint_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型的训练状态
        torch.save(model.state_dict(), self.checkpoint_name)
        self.val_loss_min = val_loss
