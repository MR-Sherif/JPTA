import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']


class Caltech101(DatasetBase):

    dataset_dir = 'Caltech101'
    alt_dataset_dirs = ('caltech-101',)

    @classmethod
    def _resolve_dataset_dir(cls, root):
        candidates = [cls.dataset_dir, *cls.alt_dataset_dirs]

        for candidate in candidates:
            dataset_dir = os.path.join(root, candidate)
            image_dir = os.path.join(dataset_dir, '101_ObjectCategories')
            split_path = os.path.join(dataset_dir, 'split_zhou_Caltech101.json')
            if os.path.isdir(image_dir) and os.path.isfile(split_path):
                return dataset_dir

        return os.path.join(root, cls.dataset_dir)

    def __init__(self, root, num_shots):
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
