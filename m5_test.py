import torch
import torch.nn as nn
from data_loader import SCLoader
from models import M5
from train import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))

# ---------- HYPERPARAMETERS ----------
batch_size = 256
new_sample_rate = 8000 
n_channel = 32
num_epochs = 120
learning_rate = 0.01 # 0.001 in original network

# brief testing shows that no class balancing converges faster. why? perhaps a flawed implementation? or n_channel is too low, net is not complex enough to benefit
# it does, however, prevent classes from having 0 predictions in the first few epochs, so it seems like its doing something right
use_class_weights = True 
# ------------------------------------

scloader = SCLoader(device, batch_size, new_sample_rate)
model = M5(n_input=scloader.n_input, n_output=len(scloader.labels))
model.to(device)
print(model)
print("Number of parameters:", model.count_params())

criterion = nn.CrossEntropyLoss()

if(use_class_weights):
    weights = scloader.get_weights()
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9) # TODO test with Adam
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

for epoch in range(1, num_epochs+1):
    train(model, scloader.transform, criterion, optimizer, scheduler, epoch, scloader.train_loader, device)
    test(model, scloader.transform, epoch, scloader.test_loader, device)
