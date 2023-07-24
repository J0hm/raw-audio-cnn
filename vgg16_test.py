import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torchaudio
from collections import OrderedDict
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))

from torchaudio.datasets import SPEECHCOMMANDS
import os

# ----- HYPERPARAMETERS -----
batch_size = 256 # might need to change this back to 256
new_sample_rate = 8000
n_channel = 64
num_epochs = 16 
learning_rate = 0.005 # 0.001 in original network
# ---------------------------



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

# this is a moderately modified implementation of torchvisions own torchvision.model.VGG16
class VGG16(nn.Module):
    def __init__(self, n_input=1, n_output=35, n_channel=32):
        super().__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv1d(n_input, n_channel, kernel_size=80, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_channel, n_channel, kernel_size=80, padding=1), # kernel = 3?
            nn.ReLU(),
            nn.MaxPool1d(4, return_indices=True),

            # conv2
            nn.Conv1d(n_channel, 2*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(2*n_channel, 2*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4, return_indices=True),

            # conv3
            nn.Conv1d(2*n_channel, 4*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(4*n_channel, 4*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(4*n_channel, 4*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4, return_indices=True),

            # conv4
            nn.Conv1d(4*n_channel, 8*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8*n_channel, 8*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8*n_channel, 8*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4, return_indices=True),

            # conv4
            nn.Conv1d(8*n_channel, 8*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8*n_channel, 8*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8*n_channel, 8*n_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4, return_indices=True)
        )

        self.classifier = nn.Sequential(
            # *7? TODO check this works for other channel sizes
            nn.Linear(8*n_channel*7, 32*n_channel),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32*n_channel, 32*n_channel),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32*n_channel, n_output)
        )

        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()

    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool1d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


model = VGG16(n_input=transformed.shape[0], n_output = len(labels), n_channel=n_channel)
model.to(device)
print(model)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_params(model)
print("Number of parameters: {}".format(n))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
    # TODO low priority fix this logging so it doesnt show errors.
    # it runs fine, but pyright doesnt like it + messy and confusing
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

# print status every log_interval iterations
log_interval = 1

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
for epoch in range(1, num_epochs + 1):
    train(model, epoch, log_interval)
    test(model, epoch)
