import os
import torch
import torchaudio
import torch.utils.data.dataloader
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm

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

class SCLoader():
    def __init__(self, device, batch_size, new_SR):
        self.__device = device
        self.__batch_size = batch_size
        self.__new_SR = new_SR
        self.__train_set = SubsetSC("training")
        self.__test_set = SubsetSC("testing")

        waveform, sample_rate, _, _, _ = self.__train_set[0]

        print("Success: grabbed dataset.")
        print("Shape of waveform: {}".format(waveform.size()))
        print("Sample rate of waveform: {}".format(sample_rate))

        self.labels = sorted(list(set(datapoint[2] for datapoint in self.__train_set)))
        print("Labels: {}".format(self.labels))

        self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.__new_SR)
        transformed = self.transform(waveform)
        self.n_input = transformed.shape[0]
        self.transform.to(device)

        print("n_input:", self.n_input)

        if device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory= False
    
        def pad_sequence(batch):
            batch = [item.t() for item in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
            return batch.permute(0, 2, 1)

        def collate_fn(batch):
            tensors, targets = [], []
    
            # encode labels as indices
            for waveform, _, label, *_ in batch:
                tensors += [waveform]
                targets += [torch.tensor(self.labels.index(label))]

            tensors = pad_sequence(tensors)
            targets = torch.stack(targets)

            return tensors, targets
    
        self.train_loader = torch.utils.data.DataLoader(
            self.__train_set,
            batch_size=self.__batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.__test_set,
            batch_size=self.__batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        return self.labels[index]

    def get_weights(self):
        print("Calculating weights...")
        weights = torch.zeros(len(self.labels), dtype=torch.long)
        weights.to(self.__device)
        for _, (_, target) in enumerate(tqdm(self.train_loader)):
            for label in target:
                weights[label] += 1

        weights = torch.max(weights)/weights 

        print("Using weights:", weights)
        return weights
