import torch.nn as nn
import torch.nn.functional as F

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

class VGG16(nn.Module):
    def __init__(self, n_input=1, n_output=35, n_channel=32, fc_channel_mul=7):
        super().__init__()

        self.features = nn.Sequential(
            vgg_conv_block([n_input,n_channel], [n_channel,n_channel], [80,3], [1,1], [1,1], 4, 4),
            vgg_conv_block([n_channel,2*n_channel], [2*n_channel,2*n_channel], [3,3], [1,1], [1,1], 4, 4),
            vgg_conv_block([2*n_channel,4*n_channel,4*n_channel], [4*n_channel,4*n_channel,4*n_channel], [3,3,3], [1,1,1], [1,1,1], 4, 4),
            vgg_conv_block([4*n_channel,8*n_channel,8*n_channel], [8*n_channel,8*n_channel,8*n_channel], [3,3,3], [1,1,1], [1,1,1], 4, 4),
            vgg_conv_block([8*n_channel,8*n_channel,8*n_channel], [8*n_channel,8*n_channel,8*n_channel], [3,3,3], [1,1,1], [1,1,1], 4, 4)
        )

        self.classifier = nn.Sequential(
            vgg_fc_layer(8*n_channel*fc_channel_mul, 64*n_channel),
            vgg_fc_layer(64*n_channel, 64*n_channel),
            nn.Linear(64*n_channel, n_output),
        )
      

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)

        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# NOTE: this is not the same network as in the paper: should be 128, 128, 256, 512
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.features = nn.Sequential(
            vgg_conv_block([n_input], [n_channel], [80], [0], [stride], 4, 4),
            vgg_conv_block([n_channel], [n_channel], [3], [0], [1], 4, 4),
            vgg_conv_block([n_channel], [2*n_channel], [3], [0], [1], 4, 4),
            vgg_conv_block([2*n_channel], [2*n_channel], [3], [0], [1], 4, 4)
        )
        
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
 
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class M11(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=4, n_channel=64):
        super().__init__()
        self.features = nn.Sequential(
            vgg_conv_block([n_input], [n_channel], [80], [38], [stride], 4, 4),
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
 
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




 



