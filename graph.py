import matplotlib.pyplot as plt
from rac import ModelManager

manager = ModelManager()

def plotModelDataFromKey(model_index, key, label=None):
    data = manager.get_model_data(model_index, key)
    x = range(len(data))
    plt.plot(x, data, label = label if label else key)

# takes a list of model indicies and a key and plots them all 
def plotModels(models, key, label_func=lambda model : manager.models[model]['model_id']):
    for model in models:
        plotModelDataFromKey(model, key, label=label_func(model))


if __name__ == '__main__':
    # params: dataset, epochs, batch_size, lr, channels, model_type
    models = manager.filter_models('sc', epochs=120)

    # keys: loss, train_accuracy, test_accuracy, precision, recall, f1
    plotModels(models, 'f1', 
               label_func=lambda model : "{}, {} channels, {} params".format(manager.models[model]['model_type'], manager.models[model]['channels'], manager.models[model]['parameters']))
    plt.legend()
    plt.show()
