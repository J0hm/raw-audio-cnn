import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from modules import visualizer
from modules.visualizer import LossVisualizer, AccuracyVisualizer
from modules.models import VGG16, M5, M11, M18
from modules.train import train, test
import modules.datasets as datasets

default_settings = {
        # [model, criterion, optimizer, n_channel, learning_rate]
        "vgg16": [VGG16, nn.CrossEntropyLoss, "SGD", 64, 0.01],
        "m5": [M5, nn.NLLLoss, "Adam", 128, 0.01],
        "m11": [M11, nn.NLLLoss, "Adam", 64, 0.01], 
        "m18": [M18, nn.NLLLoss, "Adam", 64, 0.01], 
    }

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg16", "m5", "m11", "m18"], help="Model type to train.")
parser.add_argument("dataset", choices=["sc", "marine"], help="Which dataset to train against")
parser.add_argument("epochs", help="Number of epochs to tran the model for.", type=int)
parser.add_argument("-i", "--identifier", help="Identifier when saving the model. Defaults to 'dataset'")
parser.add_argument("-l", "--learningRate", help="Starting learning rate.", type=float)
parser.add_argument("-b", "--batchSize", help="Size of each batch", type=int, default=256)
parser.add_argument("-r", "--sampleRate", help="Sample rate to resample to.", type=int, default=8000)
parser.add_argument("-c", "--channels", help="Number of channels to use.", type=int)
parser.add_argument("-p", "--patience", help="Number of epochs to wait before reducing LR on plateau", type=int, default=5)
parser.add_argument("-v", "--verbose", help="Currently unsupported.", action="store_true")

def buildOptimizer(optim_type, params, lr):
    if optim_type == "SGD":
        return optim.SGD(params, lr=lr, weight_decay=0.001, momentum=0.9)
    elif optim_type == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=0.0001)
    else:
        raise Exception("Error: invalid or unsupported optimizer type")

if __name__ == '__main__':
    args = parser.parse_args()
    defaults = default_settings[args.model]
    model_constructor, criterion, optimizer, channels, lr = defaults
    identifier = args.dataset

    if(args.channels):
        channels = args.channels
    if(args.identifier):
        identifier = args.identifier
    if(args.learningRate):
        lr = args.learningRate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    torch.set_flush_denormal(True)

    loader = datasets.supported[args.dataset](
            device=device, 
            batch_size=args.batchSize, 
            new_SR=args.sampleRate)

    model = model_constructor(
            identifier, 
            input_shape=loader.input_shape, 
            n_output=len(loader.labels),
            n_channel=channels
        )

    model.to(device)
    print(model)
    print("Number of parameters:", model.count_params())

    criterion = criterion()
    optimizer = buildOptimizer(optimizer, model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)
    loss_visualizer = LossVisualizer("{}, {}, {} channels, SR={}".format(
        args.model, 
        args.dataset,
        channels,
        args.sampleRate
    ))

    accuracy_visualizer = AccuracyVisualizer("{}, {}, {} channels, SR={}".format(
        args.model, 
        args.dataset,
        channels,
        args.sampleRate
    ))

    epoch_loss = 0
    for epoch in range(0, args.epochs):
        epoch_loss = train(model, loader.transform, criterion, optimizer, scheduler, epoch, loader.train_loader, device)
        accuracy = test(model, loader.transform, epoch, loader.test_loader, device)
        loss_visualizer.append_loss(epoch, epoch_loss)
        accuracy_visualizer.append(epoch, accuracy)

    model.save_model(args.epochs)
