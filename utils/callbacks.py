import os
import scipy.signal
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def append_data(self, epoch_loss, epoch_acc, phase):
        if phase == 'train':
            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)
            with open(os.path.join(self.log_dir, phase + "_loss" + ".txt"), 'a') as f:
                f.write(str(epoch_loss))
                f.write("\n")
            with open(os.path.join(self.log_dir, phase + "acc_" + ".txt"), 'a') as f:
                f.write(str(epoch_acc))
                f.write("\n")
            self.loss_plot()
        else:
            self.val_loss.append(epoch_loss)
            self.val_acc.append(epoch_acc)
            with open(os.path.join(self.log_dir, phase + "_loss" + ".txt"), 'a') as f:
                f.write(str(epoch_loss))
                f.write("\n")
            with open(os.path.join(self.log_dir, phase + "acc_" + ".txt"), 'a') as f:
                f.write(str(epoch_acc))
                f.write("\n")
            self.loss_plot()
            self.acc_plot()

    def loss_plot(self):
        iterx = range(len(self.train_loss))
        itery = range(len(self.val_loss))
        plt.figure()
        plt.plot(iterx, self.train_loss, 'red', linewidth=2, label='train loss')
        plt.plot(itery, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss_" + ".png"))
        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iterx = range(len(self.train_acc))
        itery = range(len(self.val_acc))
        plt.figure()
        plt.plot(iterx, self.train_acc, 'red', linewidth=2, label='train acc')
        plt.plot(itery, self.val_acc, 'coral', linewidth=2, label='val acc')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_acc_" + ".png"))
        plt.cla()
        plt.close("all")
