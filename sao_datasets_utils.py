from pathlib import Path

import torch
from torch.utils.data import Dataset
from random import seed, sample

class AlignedImageFeaturesDataset(Dataset):
    def __init__(self, image_folder_dataset, features_dataset):
        self.image_folder_dataset = image_folder_dataset
        self.features_dataset = features_dataset

        assert len(self.image_folder_dataset) == len(self.features_dataset)

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, i):
        img_path = Path(self.image_folder_dataset.imgs[i][0]).stem + ".bmp"
        return {
            "image": self.image_folder_dataset[i][0],
            "features": self.features_dataset[img_path]["features"],
            "label": self.image_folder_dataset[i][1],
        }
    


class ZOISDatasetFeatures(Dataset):
    def __init__(
        self,
        data_path: Path,
        text_useful_features: int = -1,
        per_class_max=None,
        seed_n=None,
    ):
        Dataset.__init__(self)
        (
            self.labels,
            self.label2idx,
            self.text_data,
            self.labels_data,
            self.path2index,
        ) = self._load_data(data_path, text_useful_features, seed_n, per_class_max)

    def __getitem__(self, i):
        if isinstance(i, str):
            i = self.path2index[i]

        return {
            "features": self.text_data[i],
            "label": self.labels_data[i],
        }

    def __len__(self):
        return len(self.text_data)

    def normalize1D(self, data):
        stack = torch.stack(data)
        mean, std = stack.mean(dim=0), stack.std(dim=0)
        std[std == 0] = 1
        return [(x - mean) / std for x in data]

    def _keep_positions(self, l, positions):
        keep = []
        for pos in positions:
            if isinstance(pos, int):
                keep.append(l[pos])
            elif isinstance(pos, tuple):
                keep.extend(l[pos[0] : pos[1]])
            else:
                raise ValueError(
                    f"Invalid value {pos} for `_keep_positions` 'positions' argument"
                )
        return keep

    def _load_data(
        self, data_path: Path, useful_features: int, seed_n: int, per_class_max: int
    ):
        if seed_n is not None:
            seed(seed_n)

        labels = []
        label2indx = {}
        text_data = []
        labels_data = []
        path2index = {}
        for i, folder in enumerate(data_path.iterdir()):
            class_name = folder.name
            print(f"Reading '{class_name}' files")
            labels.append(class_name)
            label2indx[class_name] = i

            cls_data = []
            cls_labels = []
            cls_paths = []

            # --- READING .data FILE
            data_file_path = folder / f"{class_name}.data"
            if data_file_path.exists():
                lines = data_file_path.read_text().splitlines()

                for line in lines[3:]:
                    line = line.split()
                    cls_paths.append(line[-8])
                    line = self._keep_positions(line, useful_features)
                    cls_data.append(torch.tensor(list(map(float, line))))
                    cls_labels.append(i)

            if per_class_max is not None:
                n = len(cls_data)
                if n > per_class_max:
                    indexes = sample(range(n), per_class_max)
                    cls_data = [cls_data[i] for i in indexes]
                    cls_labels = [cls_labels[i] for i in indexes]
                    cls_paths = [cls_paths[i] for i in indexes]

            text_data.extend(cls_data)
            labels_data.extend(cls_labels)

            for path in cls_paths:
                path2index[path] = len(path2index)

        # --- NORMALIZING DATA
        text_data = self.normalize1D(text_data)

        return labels, label2indx, text_data, labels_data, path2index

