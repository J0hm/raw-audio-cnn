import torch
import torch.nn as nn
from data_loader import SCLoader
from visualizer import LossVisualizer
from models import M11
from train import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))
torch.set_flush_denormal(True)

# ---------- HYPERPARAMETERS ----------
batch_size = 256
new_sample_rate = 8000 
n_channel = 64
num_epochs = 120
learning_rate = 0.01 
# ------------------------------------

scloader = SCLoader(device, batch_size, new_sample_rate)
n_output = len(scloader.labels)
model = M11(n_input=scloader.n_input, n_output=n_output, n_channel=n_channel)
model.to(device)
print(model)
print("Number of parameters:", model.count_params())

criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001) # TODO test with Adam
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
visualizer = LossVisualizer("M11 Training Loss")

for epoch in range(0, num_epochs):
    epoch_loss = train(model, scloader.transform, criterion, optimizer, scheduler, epoch, scloader.train_loader, device)
    visualizer.append_loss(epoch, epoch_loss)
    test(model, scloader.transform, epoch, scloader.test_loader, device)
