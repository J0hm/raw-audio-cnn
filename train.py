import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from rac import VGG16, M5, M11, M18, train, test, TrainDataManager, datasets

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
parser.add_argument("-v", "--verbose", help="Enables extra logging.", action="store_true")
parser.add_argument("-v2", "--verbose2", help="Enables extra logging and visdom.", action="store_true")


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
    identifier = "*"

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
            args.dataset, identifier,
            input_shape=loader.input_shape, 
            n_output=len(loader.labels),
            n_channel=channels
        )

    model.to(device)
    
    manager = TrainDataManager(model, lr, args.batchSize, args.epochs)

    criterion = criterion()
    optimizer = buildOptimizer(optimizer, model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)

    epoch_loss = 0
    for epoch in range(0, args.epochs):
        epoch_loss = train(model, loader.transform, criterion, optimizer, scheduler, epoch, loader.train_loader, device)
        accuracy_test, macros = test(model, loader.transform, epoch, loader.test_loader, device, verbose=args.verbose, labels=loader.labels)
        accuracy_train, _ = test(model, loader.transform, epoch, loader.train_loader, device, verbose=args.verbose)

        manager.append_epoch(epoch, epoch_loss, accuracy_train, accuracy_test, macros['precision'], macros['recall'], macros['f1-score'])

    manager.save_model()
