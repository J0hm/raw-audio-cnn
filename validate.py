import torch
import argparse
from modules.models import loadModel
from modules.train import test

#infer and batch modes
#parser = argparse.ArgumentParser()
#parser.add_argument("model", help="Model to validate")
#parser.add_argument("-b", "--batchSize", help="Size of each batch", type=int, default=256)
#parser.add_argument("-r", "--sampleRate", help="Sample rate to resample to.", type=int, default=8000)

if __name__ == '__main__':
    model_name = "m5_marine_128_60.pt"
    batch_size = 256
    new_sample_rate = 8000 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    torch.set_flush_denormal(True)

    model, loader = loadModel(model_name, batch_size, new_sample_rate, device)
    print("Model {} loaded.".format(model_name))

    acc = test(model, loader.transform, "VALIDATE", loader.validate_loader, device, verbose=True, labels=loader.labels)
