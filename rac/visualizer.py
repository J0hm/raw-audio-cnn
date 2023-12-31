import visdom
import torch

class LossVisualizer():
    def __init__(self, title):
        self.__vis = visdom.Visdom()
        self.__loss_window = self.__vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Epoch',ylabel='Loss',title=title,legend=['Loss']))

    def append(self, epoch, epoch_loss):
        self.__vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([epoch_loss]).unsqueeze(0).cpu(),win=self.__loss_window,update='append')


        
class AccuracyVisualizer():
    def __init__(self, title):
        self.__vis = visdom.Visdom()
        self.__loss_window = self.__vis.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Epoch',ylabel='Accuracy',title=title,legend=['Accuracy']))

    def append(self, epoch, epoch_accuracy):
        self.__vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([epoch_accuracy]).unsqueeze(0).cpu(),win=self.__loss_window,update='append')
