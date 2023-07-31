import torch
from model_loader import loadModel
from train import test


if __name__ == '__main__':
    model_name = "m11_marine_64_120.pt"
    batch_size = 256
    new_sample_rate = 8000 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    torch.set_flush_denormal(True)

    model, loader = loadModel(model_name, batch_size, new_sample_rate, device)
    print("Model {} loaded.".format(model_name))

    test(model, loader.transform, "VALIDATE", loader.validate_loader, device, verbose=True)
