import random
import string
import json
import os
import errno
from torchinfo import summary
from modules.models import loadModel
from csv import writer

alphabet = string.ascii_lowercase + string.digits

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


# Manages training statistics
class TrainDataManager:
    def __init__(self, model, lr, batch_size, epochs):
        self.model = model
        self.id = ''.join(random.choices(alphabet, k=8))
        self.name = "{}_{}_{}_{}_{}".format(model.modeltype, model.dataset, model.identifier, epochs, self.id)

        self.path = "models/{}".format(self.id)
        mkdir_p(self.path)

        self.gen_metadata_file(lr, batch_size, epochs)
        self.gen_summary_file()

       


    def gen_metadata_file(self, lr, batch_size, epochs):
        #include: modeltype, dataset, param ct, channels, lr, batch size, epochs
        metadata = {
                "model_name": self.name,
                "model_type": self.model.modeltype,
                "dataset": self.model.dataset,
                "parameters": self.model.count_params(),
                "channels": self.model.n_channel,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs}

        json_obj = json.dumps(metadata, indent=4)
        
        with open(os.path.join(self.path, "metadata.json"), 'w') as out:
            out.write(json_obj)

    def gen_summary_file(self):
        with open(os.path.join(self.path, "summary.txt"), 'w') as f:
            print(self.model.in_shape)
            model_stats = summary(self.model, col_names=("num_params", "kernel_size")) 
            f.write(str(model_stats))

    def append_epoch(self, epoch, loss, train_accuracy, test_accuracy,
                     precision, recall, f1):
        with open(os.path.join(self.path, "trainstats.csv"), 'a+') as csv_file:
            w = writer(csv_file)
            w.writerow([epoch, loss, train_accuracy, test_accuracy, precision, recall, f1])

    def save_model(self):
        self.model.save_model(os.path.join(self.path, "model.pt"))



class ModelManager:
    def __init__(self, model_directory="models"):
        self.models = self.init_models(model_directory)

    def init_models(self, model_directory):
        res = {}
        for idx, subdir in enumerate(os.listdir(model_directory)):
            model_path = os.path.join(model_directory, subdir, "model.pt")
            with open(os.path.join(model_directory, subdir, "metadata.json")) as data_file:
                metadata = json.load(data_file)
                metadata['model_path'] = model_path
                res[idx] = metadata

        return res

    def __str__(self):
        return json.dumps(self.models, indent=2)

    def load_model(self, index, sample_rate, device):
        metadata = self.models[index]
        return loadModel(
                metadata['model_path'], 
                metadata['model_type'],
                metadata['dataset'],
                metadata['batch_size'],
                sample_rate,
                metadata['channels'],
                device)


