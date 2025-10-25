from algorithms.mDSDI.src.dataloaders.MNIST_Dataloader import MNIST_Test_Dataloader, MNISTDataloader
from algorithms.mDSDI.src.dataloaders.Standard_Dataloader import StandardDataloader, StandardValDataloader
from algorithms.mDSDI.src.dataloaders.BCI2a_Dataloader import BCI2a_Train_Dataset, BCI2a_Test_Dataset


train_dataloaders_map = {
    "PACS": StandardDataloader,
    "DomainNet": StandardDataloader,
    "MNIST": MNISTDataloader,
    "OfficeHome": StandardDataloader,
    "VLCS": StandardDataloader,
    "BCI2a": BCI2a_Train_Dataset,
}

test_dataloaders_map = {
    "PACS": StandardValDataloader,
    "DomainNet": StandardValDataloader,
    "MNIST": MNIST_Test_Dataloader,
    "OfficeHome": StandardValDataloader,
    "VLCS": StandardValDataloader,
    "BCI2a": BCI2a_Test_Dataset,
}


def get_train_dataloader(name):
    if name not in train_dataloaders_map:
        raise ValueError("Name of train dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return train_dataloaders_map[name](**kwargs)

    return get_dataloader_fn


def get_test_dataloader(name):
    if name not in test_dataloaders_map:
        raise ValueError("Name of test dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return test_dataloaders_map[name](**kwargs)

    return get_dataloader_fn
