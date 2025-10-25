from algorithms.mDSDI.src.models.mnistnet import MNIST_CNN, Color_MNIST_CNN
from algorithms.mDSDI.src.models.resnet import ResNet
from algorithms.mDSDI.src.models.eeg_models import EEG_CNN, EEG_ResNet
from algorithms.mDSDI.src.models.eeg_models_advanced import EEG_CNN_Optimized, EEG_CNN_Attention


nets_map = {
    "mnistnet": MNIST_CNN, 
    "cmnistnet": Color_MNIST_CNN, 
    "resnet50": ResNet,
    "resnet50_bci2a": lambda: ResNet(input_channels=22),
    "eeg_cnn": EEG_CNN,
    "eeg_resnet": EEG_ResNet,
    "eeg_cnn_optimized": EEG_CNN_Optimized,
    "eeg_cnn_attention": EEG_CNN_Attention,
}


def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn
