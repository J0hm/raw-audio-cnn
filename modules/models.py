import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.data_loader import SCLoader, MammalLoader


def conv_layer(chann_in, chann_out, k_size, p_size, stride):
    layer = nn.Sequential(
        nn.Conv1d(chann_in, chann_out, kernel_size=k_size, padding=p_size, stride=stride),
        nn.BatchNorm1d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, s_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i], s_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool1d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out, dropout=True):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU(),
    )
    if(dropout):
        layer.append(nn.Dropout())

    return layer



class AbstractModel(nn.Module):
    def __init__(self, modeltype, identifier, n_channel):
        super(AbstractModel, self).__init__()
        self.modeltype = modeltype
        self.n_channel = n_channel
        self.identifier = identifier

    def save_model(self, epoch=None, path="models"):
        name = "{}_{}_{}.pt".format(self.modeltype, self.identifier, self.n_channel)
        if(epoch):
            name = "{}_{}_{}_{}.pt".format(self.modeltype, self.identifier, self.n_channel, epoch)

        torch.save(self.state_dict(), os.path.join(path, name))

    def save_checkpoint(self, optimizer, epoch, loss, path="models"):
        name = "{}_{}_{}_{}_ckpt.pt".format(self.modeltype, self.identifier, self.n_channel, epoch)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(path, name))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class VGG16(AbstractModel):
    def __init__(self, identifier, input_shape, n_output=35, n_channel=64):
        super().__init__("vgg16", identifier, n_channel)
        self.fc_channel_mul = round((input_shape[1]+1)/1000)-1
        self.features = nn.Sequential(
            vgg_conv_block([input_shape[0],n_channel], [n_channel,n_channel], [80,3], [1,1], [1,1], 4, 4),
            vgg_conv_block([n_channel,2*n_channel], [2*n_channel,2*n_channel], [3,3], [1,1], [1,1], 4, 4),
            vgg_conv_block([2*n_channel,4*n_channel,4*n_channel], [4*n_channel,4*n_channel,4*n_channel], [3,3,3], [1,1,1], [1,1,1], 4, 4),
            vgg_conv_block([4*n_channel,8*n_channel,8*n_channel], [8*n_channel,8*n_channel,8*n_channel], [3,3,3], [1,1,1], [1,1,1], 4, 4),
            vgg_conv_block([8*n_channel,8*n_channel,8*n_channel], [8*n_channel,8*n_channel,8*n_channel], [3,3,3], [1,1,1], [1,1,1], 4, 4)
        )

        self.classifier = nn.Sequential(
            vgg_fc_layer(8*n_channel*self.fc_channel_mul, 64*n_channel),
            vgg_fc_layer(64*n_channel, 64*n_channel),
            nn.Linear(64*n_channel, n_output),
        )
      

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)

        return out


class M5(AbstractModel):
    def __init__(self, identifier, input_shape, n_output=35, stride=4, n_channel=128):
        super().__init__("m5", identifier, n_channel)

        self.features = nn.Sequential(
            vgg_conv_block([input_shape[0]], [n_channel], [80], [38], [stride], 4, 4),
            vgg_conv_block([n_channel], [n_channel], [3], [1], [1], 4, 4),
            vgg_conv_block([n_channel], [2*n_channel], [3], [1], [1], 4, 4),
            vgg_conv_block([2*n_channel], [4*n_channel], [3], [1], [1], 4, 4)
        )
        
        self.fc1 = nn.Linear(4 * n_channel, n_output)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M11(AbstractModel):
    def __init__(self, identifier, input_shape, n_output=35, stride=4, n_channel=64):
        super().__init__("m11", identifier, n_channel) 
        self.features = nn.Sequential(
            vgg_conv_block([input_shape[0]], [n_channel], [80], [38], [stride], 4, 4),
            vgg_conv_block([n_channel, n_channel], [n_channel, n_channel], [3, 3], [1, 1], [1, 1], 4, 4),
            vgg_conv_block([n_channel, 2*n_channel], [2*n_channel, 2*n_channel], [3, 3], [1, 1], [1, 1], 4, 4),
            vgg_conv_block([2*n_channel, 4*n_channel, 4*n_channel], [4*n_channel, 4*n_channel, 4*n_channel], [3, 3, 3], [1, 1, 1], [1, 1, 1], 4, 4),
            vgg_conv_block([4*n_channel, 8*n_channel], [8*n_channel, 8*n_channel], [3, 3], [1, 1], [1, 1], 4, 4)
        )
        
        self.fc1 = nn.Linear(8*n_channel, n_output)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M18(AbstractModel):
    def __init__(self, identifier, input_shape, n_output=35, stride=4, n_channel=64):
        super().__init__("m18", identifier, n_channel)
        self.features = nn.Sequential(
            vgg_conv_block([input_shape[0]], [n_channel], [80], [38], [stride], 4, 4),
            vgg_conv_block([n_channel]*4, [n_channel]*4, [3]*4, [1]*4, [1]*4, 4, 4),
            vgg_conv_block([n_channel, 2*n_channel, 2*n_channel, 2*n_channel], [2*n_channel]*4, [3]*4, [1]*4, [1]*4, 4, 4),
            vgg_conv_block([2*n_channel, 4*n_channel, 4*n_channel, 4*n_channel], [4*n_channel]*4, [3]*4, [1]*4, [1]*4, 4, 4),
            vgg_conv_block([4*n_channel, 8*n_channel, 8*n_channel, 8*n_channel], [8*n_channel]*4, [3]*4, [1]*4, [1]*4, 4, 4),
        )
        
        self.fc1 = nn.Linear(8*n_channel, n_output)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
 


models = {
        "m5": M5,
        "m11": M11,
        "m18": M18,
        "vgg16": VGG16
    }

loaders = {
        "sc": SCLoader,
        "marine": MammalLoader
    }



def loadModel(model_name, batch_size, sample_rate, device, model_folder="models"):
    path = os.path.join(model_folder, model_name)
    params = model_name.split("_")
    
    print(params)

    # constrct the data loader from the given parameters
    data = loaders[params[1]](device, batch_size, sample_rate)

    # build the model from the appropriate constructor,
    # using the parameters stored in the model name
    model = models[params[0]](
            params[1],
            input_shape=data.input_shape,
            n_output=len(data.labels),
            n_channel=int(params[2])
        )

    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    return (model, data)

