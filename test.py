from modules.data_management import ModelManager


manager = ModelManager()

print(manager.filter_models("marine", model_type="m5", epochs=60))
