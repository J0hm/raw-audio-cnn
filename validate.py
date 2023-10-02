import torch
import argparse
import sys
from rac import ModelManager, test

#infer and batch modes
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Index of the model to validate. 'list' to list all models")
parser.add_argument("-r", "--sampleRate", help="Sample rate to resample to.", type=int, default=8000)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    torch.set_flush_denormal(True)
    
    manager = ModelManager()

    if(args.model == 'list'):
        print(str(manager))
        sys.exit()

    model, loader = manager.load_model(int(args.model), args.sampleRate, device)

    print("Model {} loaded.".format(args.model))

    acc = test(model, loader.transform, "VALIDATE", loader.validate_loader, device, verbose=True, labels=loader.labels)
