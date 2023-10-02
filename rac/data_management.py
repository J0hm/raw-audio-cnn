import random
import string
import json
import os
import errno
from torchinfo import summary
from .models import loadModel
from csv import reader, writer

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
        self.train_data_keys = ['loss', 'train_accuracy', 'test_accuracy', 'precision', 'recall', 'f1']

    def init_models(self, model_directory):
        res = {}
        for idx, subdir in enumerate(os.listdir(model_directory)):
            model_path = os.path.join(model_directory, subdir)
            with open(os.path.join(model_directory, subdir, "metadata.json")) as data_file:
                metadata = json.load(data_file)
                metadata['model_path'] = model_path
                metadata['model_id'] = subdir
                res[idx] = metadata

        return res

    def __str__(self):
        return json.dumps(self.models, indent=2)

    def load_model(self, index, sample_rate, device):
        metadata = self.models[index]
        return loadModel(
                os.path.join(metadata['model_path'], "model.pt"), 
                metadata['model_type'],
                metadata['dataset'],
                metadata['batch_size'],
                sample_rate,
                metadata['channels'],
                device)
    
    def get_key_index(self, key):
        return self.train_data_keys.index(key)+1

    # returns the column for key
    def get_model_data(self, model_index, key="loss"):
        res = []
        with open(os.path.join(self.models[model_index]['model_path'], "trainstats.csv")) as csv_file:
            r = reader(csv_file)
            for row in r:
                data = float(row[self.get_key_index(key)])
                res.append(data)

        return res

    # returns a list of indicies for filtered models
    # parameters set to none are ignored
    def filter_models(self, dataset, epochs=None, batch_size=None, lr=None, channels=None, model_type=None):
        def f(entry):
            _, data = entry
            if(data['dataset'] != dataset):
                return False
            elif(epochs and data['epochs'] != epochs):
                return False
            elif(batch_size and data['batch_size'] != batch_size):
                return False
            elif(lr and data['lr'] != lr):
                return False
            elif(channels and data['channels'] != channels):
                return False
            elif(model_type and data['model_type'] != model_type):
                return False

            return True

        return [idx for idx, _ in filter(f, self.models.items())]

