from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT

dataset = DREAMERDataset(
    mat_path="emotion_data/Dreamer/DREAMER.mat",
    # offline_transform=transforms.Compose(
    #     [
    #         transforms.BandDifferentialEntropy(),
    #         transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT),
    #     ]
    # ),
    # online_transform=transforms.ToTensor(),
    # label_transform=transforms.Compose(
    #     [
    #         transforms.Select("valence"),
    #         transforms.Binary(3.0),
    #     ]
    # ),
)
print(dataset[1])
