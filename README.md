# Various CNN architecture implementations for raw signal/audio classification
You MUST run `python3 -m venv .venv`, `. start_venv.sh`, and then `python3 -m pip install -r requirements.txt` before using any of the scripts or models in this repository. If you wish to use Visdom for data visualization, run `visdom` in the venv.

Currently, M5, M11, M18 and VGG16 are supported, with more on the way.

`train` and `validate` are fully implemented, with a visualizer partially implemented.

## Main modules
### train
Simple CLI implementation to train various model types against the included datasets. Hyperparameters may be tuned.

### validate
Simple CLI implementation to validate different trained models against the validation set. `python3 -m validate list` to list models and `python3 -m validate {idx}` to validate the model with the given index.

### graph
Not yet fully implemented, but allows for filtering model by metadata and then graphing the desired parameter.

## Submodules
### models.py
Contains model implementation and a simple API for easilly implementing new models.

### train.py
Contains the train and test functions.

### datasets.py
Contains information about the implemented datasets for easy querying and dynamic selection. Should probably be moved to `data_loader.py` when the dataloader is refactored.

### visualizer.py
Contains a class for visualizing losses and accuracy with Visdom. Not used by default (unless train is run with the flag `-v2`).

### data_loader.py
Contains implementations for the two currently supported datasets and an abstract class to be used as an interface to implement more. This module needs to be refactored to make it safer and easier to work with.

### data_management.py
Contains classes for writing train output data and then reading and listing information about the trained models.
