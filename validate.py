import torch
import argparse
from modules.models import loadModel, loadModelInfer
from modules.train import test

#infer and batch modes
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model to validate.")
parser.add_argument("-i", "--infer", help="Infer parameters from the model name", action="store_true")
parser.add_argument("-m", "--modelType", choices=["vgg16", "m5", "m11", "m18"], help="Model architecture of the model to load.")
parser.add_argument("-d", "--dataset", choices=["sc", "marine"], help="Which dataset to train against")
parser.add_argument("-b", "--batchSize", help="Size of each batch", type=int, default=256)
parser.add_argument("-r", "--sampleRate", help="Sample rate to resample to.", type=int, default=8000)
parser.add_argument("-c", "--channels", help="Number of channels of the loaded model", type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    torch.set_flush_denormal(True)

    if(args.infer):
        print("Inferring model params...")
        model, loader = loadModelInfer(args.model, args.batchSize, args.sampleRate, device)
    else:
        model, loader = loadModel(args.model, args.modelType, args.dataset, args.batchSize, args.sampleRate, args.channels, device)

    print("Model {} loaded.".format(args.model))

    acc = test(model, loader.transform, "VALIDATE", loader.validate_loader, device, verbose=True, labels=loader.labels)
