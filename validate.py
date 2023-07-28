import torch
from models import M5
from data_loader import SCLoader
from train import test

path = "saved/m5_128_60.pt"
batch_size = 256
new_sample_rate = 8000 
n_channel = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))
torch.set_flush_denormal(True)

scloader = SCLoader(device, batch_size, new_sample_rate)
n_output = len(scloader.labels)
model = M5(n_input=scloader.n_input, n_output=n_output, n_channel=n_channel)
model.load_state_dict(torch.load(path, map_location=torch.device(device)))
model.eval()
print("Model {} loaded.".format(path))

test(model, scloader.transform, "VALIDATE", scloader.validate_loader, device, verbose=True)
