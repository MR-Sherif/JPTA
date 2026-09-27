import torch
import torchvision.transforms as transforms
from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .utils import *

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                }


def get_all_dataloaders(args, preprocess, num_workers = 8):
    dataset_name = args.dataset
    train_loader = None
    val_loader = None
    sampler = None

    if dataset_name.startswith('imagenet'):
        load_cache = getattr(args, 'load_cache', True)
        load_pre_feat = getattr(args, 'load_pre_feat', False)
        dataset = dataset_list[dataset_name](
            args.root_path,
            0,
            preprocess=preprocess,
            train_preprocess=None,
            test_preprocess=None,
            load_cache=load_cache,
            load_pre_feat=load_pre_feat,
        )
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=num_workers, shuffle=False, sampler=sampler)

    else:
        dataset = dataset_list[dataset_name](args.root_path, 0)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                       shuffle=False, num_workers = num_workers)

        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                        shuffle=False, sampler=sampler, num_workers = num_workers)

    return train_loader, val_loader, test_loader, dataset
