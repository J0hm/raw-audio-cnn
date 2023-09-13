import os
import torch
import torchaudio.transforms
import torchaudio
import torch.utils.data.dataloader
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm

# TODO: abstract class for dataloaders

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
        self.__validate_set = SubsetSC("validation")

        waveform, sample_rate, _, _, _ = self.__train_set[0]

        print("Success: grabbed dataset.")

        self.labels = sorted(list(set(datapoint[2] for datapoint in self.__train_set)))
        print("Labels: {}".format(self.labels))

        self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.__new_SR)
        transformed = self.transform(waveform)
        self.input_shape = transformed.shape
        self.transform.to(device)

        print("Input shape: {}, {} sps".format(self.input_shape, new_SR))


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
        self.validate_loader = torch.utils.data.DataLoader(
            self.__validate_set,
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

class MarineMammalDataset(Dataset):
    def __init__(self, csv_file, root_dir, pad_to=2, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.pad_to = pad_to

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx, 1]


        waveform, sample_rate = torchaudio.load(wav_name)
        num_frames = torchaudio.info(wav_name).num_frames
        req_frames = self.pad_to*sample_rate
        if(req_frames - num_frames > 0):
            waveform = F.pad(input=waveform, pad=(0, req_frames-num_frames), mode='constant', value=0)
        
        if self.transform:
            waveform = self.transform(waveform)

        return (waveform, label)

class MammalLoader():
    def __init__(self, device, batch_size, new_SR, pad_to=1, data_path="marineaudio/datasets/len1/"):
        self.labels = ['Balaena_mysticetus', 'Balaenoptera_physalus', 'Delphinapterus_leucas', 'Delphinus_delphis', 'Erignathus_barbatus', 'Eubalaena_australis', 'Eubalaena_glacialis', 'Globicephala_macrorhynchus', 'Globicephala_melas', 'Grampus_griseus', 'Hydrurga_leptonyx', 'Lagenodelphis_hosei', 'Lagenorhynchus_acutus', 'Lagenorhynchus_albirostris', 'Megaptera_novaeangliae', 'Monodon_monoceros', 'Odobenus_rosmarus', 'Ommatophoca_rossi', 'Orcinus_orca', 'Pagophilus_groenlandicus', 'Peponocephala_electra', 'Physeter_macrocephalus', 'Pseudorca_crassidens', 'Stenella_attenuata', 'Stenella_clymene', 'Stenella_coeruleoalba', 'Stenella_frontalis', 'Stenella_longirostris', 'Steno_bredanensis', 'Tursiops_truncatus']
        self.__device = device
        self.__batch_size = batch_size
        self.__train_set = MarineMammalDataset(os.path.join(data_path, "train.csv"), 
                                               os.path.join(data_path, "audio"), 
                                               pad_to=pad_to)
        self.__test_set = MarineMammalDataset(os.path.join(data_path, "test.csv"), 
                                               os.path.join(data_path, "audio"), 
                                               pad_to=pad_to)
        self.__validate_set = MarineMammalDataset(os.path.join(data_path, "validate.csv"), 
                                               os.path.join(data_path, "audio"), 
                                               pad_to=pad_to)

        print("Success: grabbed dataset.")
        print("Labels: {}".format(self.labels))
        waveform, _ = self.__train_set.__getitem__(0)
        self.transform = torchaudio.transforms.Resample(orig_freq=22050, new_freq=new_SR)
        self.transform.to(device)
        transformed = self.transform(waveform)
        self.input_shape = transformed.shape
        print("Input shape: {}, {} sps".format(self.input_shape, new_SR))

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
            for (waveform, label) in batch:
                tensors += [waveform]
                targets += [torch.tensor(label)] 
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
        self.validate_loader = torch.utils.data.DataLoader(
            self.__validate_set,
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
        
        print(weights, torch.sum(weights))
        weights = torch.max(weights)/weights 

        print("Using weights:", weights)
        return weights
