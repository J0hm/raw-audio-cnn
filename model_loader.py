from models import VGG16, M5, M11
from data_loader import SCLoader, MammalLoader
import torch
import os

models = {
        "m5": M5,
        "m11": M11,
        "vgg16": VGG16
    }

loaders = {
        "sc": SCLoader,
        "marine": MammalLoader
    }

def loadModel(model_name, batch_size, sample_rate, device, model_folder="models"):
    path = os.path.join(model_folder, model_name)
    params = model_name.split("_")
    
    print(params)

    # constrct the data loader from the given parameters
    data = loaders[params[1]](device, batch_size, sample_rate)

    # build the model from the appropriate constructor,
    # using the parameters stored in the model name
    model = models[params[0]](
            n_input=data.n_input,
            n_output=len(data.labels),
            n_channel=int(params[2])
        )

    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    return (model, data)




