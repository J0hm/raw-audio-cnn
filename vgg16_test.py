import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torchaudio
import numpy
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))

from torchaudio.datasets import SPEECHCOMMANDS
import os

# ---------- HYPERPARAMETERS ----------
batch_size = 256
num_classes = 35
new_sample_rate = 8000 # BE VERY CAREFUL CHANGING THIS. Input size changes FC_CHANNEL_MUL must be cchanged too
FC_CHANNEL_MUL = 7 # SR->MUL: 2000->1, 4000->3, 8000->7, 16000->15. 2*SR->2*MUL(SR)+1
n_channel = 64
num_epochs = 120
learning_rate = 0.01 # 0.001 in original network

# brief testing shows that no class balancing converges faster. why? perhaps a flawed implementation? or n_channel is too low, net is not complex enough to benefit
# it does, however, prevent classes from having 0 predictions in the first few epochs, so it seems like its doing something right
use_class_weights = True 
# ------------------------------------


# -----------------------------
# ----- BEGIN DATA SECTION ----
# -----------------------------
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

print("Success: grabbed dataset.")
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
print("Labels: {}".format(labels))

transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

def label_to_index(word):
    return torch.tensor(labels.index(word))

def index_to_label(index):
    return labels[index]

def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    tensors, targets = [], []

    # encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def get_weights():
    print("Calculating weights...")
    weights = torch.zeros(num_classes, dtype=torch.long)
    weights.to(device)
    for _, (_, target) in enumerate(tqdm(train_loader)):
        for label in target:
            weights[label] += 1

    weights = torch.max(weights)/weights 

    print("Using weights:", weights)
    return weights


if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory= False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory
)



# -------------------------------
# ----- BEGIN TRAIN SECTION -----
# -------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv1d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm1d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool1d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU(),
        nn.Dropout() # TODO dropout testing
    )

    return layer

# this is a moderately modified implementation of torchvisions own torchvision.model.VGG16
class VGG16(nn.Module):
    def __init__(self, n_input=1, n_output=35, n_channel=32):
        super().__init__()

        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([n_input,n_channel], [n_channel,n_channel], [80,3], [1,1], 4, 4)
        self.layer2 = vgg_conv_block([n_channel,2*n_channel], [2*n_channel,2*n_channel], [3,3], [1,1], 4, 4)
        self.layer3 = vgg_conv_block([2*n_channel,4*n_channel,4*n_channel], [4*n_channel,4*n_channel,4*n_channel], [3,3,3], [1,1,1], 4, 4)
        self.layer4 = vgg_conv_block([4*n_channel,8*n_channel,8*n_channel], [8*n_channel,8*n_channel,8*n_channel], [3,3,3], [1,1,1], 4, 4)
        self.layer5 = vgg_conv_block([8*n_channel,8*n_channel,8*n_channel], [8*n_channel,8*n_channel,8*n_channel], [3,3,3], [1,1,1], 4, 4)

        # FC layers
        self.layer6 = vgg_fc_layer(8*n_channel*FC_CHANNEL_MUL, 64*n_channel)
        self.layer7 = vgg_fc_layer(64*n_channel, 64*n_channel)

        # Final layer
        self.layer8 = nn.Linear(64*n_channel, n_output)
      

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        # we change the view here to have the output be the right shape for linear layers
        # something like [256, 256, 7] -> [256, 1792] i.e. batch of size 256 with 256 channels of 7 features each to batch of size 256 with 1792 features each
        out = out.view(out.size()[0], -1)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


model = VGG16(n_input=transformed.shape[0], n_output = len(labels), n_channel=n_channel)
model.to(device)
print(model)

n = count_params(model)
print("Number of parameters: {}".format(n))

criterion = nn.CrossEntropyLoss()

if(use_class_weights):
    weights = get_weights()
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9) # TODO test with Adam
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

def train(model, epoch, verbose=False, log_interval=10):
    model.train()

    total_loss = 0
    count = 0
    
    print("Beginning epoch", epoch, "...")

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        loss = criterion(output.squeeze(), target) # is squeeze necessary? might not be if there are no dimensions of size 1
        total_loss += loss.item()
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if verbose and batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.1f}%)]\tLoss: {loss.item():.6f}\tAvg loss: {total_loss / count:.6f}")

    print(f"Epoch: {epoch} completed\tAvg loss: {total_loss / count:.6f}")
    scheduler.step(total_loss)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(model, epoch, verbose=False):
    model.eval()
    correct = 0
    counts_pred = numpy.zeros(35, dtype = int)
    counts_actual = numpy.zeros(35, dtype = int)

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        
        for l in target:
            counts_actual[l] += 1

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        for p in pred:
            counts_pred[p] += 1

    if(verbose):
        print("Predicted label counts:\n", counts_pred)
        print("Actual label counts:\n", counts_actual)
        print("Diff:\n", (counts_pred-counts_actual))

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
for epoch in range(1, num_epochs + 1):
    train(model, epoch)
    test(model, epoch)
    
