from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT
import numpy as np
import pandas as pd

dataset = DREAMERDataset(
    mat_path="emotion_data\Dreamer\DREAMER.mat",
    online_transform=transforms.To2d(),
    label_transform=None,
    io_path=".torcheeg\datasets_1768356149428_vf5Ep",
    num_worker=4,
)
# print(dataset[1])


rows = []
for idx in range(len(dataset)):
    eeg, shape, dic = dataset[idx]  # adjust if your print(dataset[0]) shows a different structure
    # eeg: (n_channels, n_points)
    eeg = np.array(eeg)
    flat = eeg.flatten()
    row = {"sample_idx": idx}
    for i, v in enumerate(flat):
        row[f"eeg_{i}"] = float(v)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("dreamer_eeg_flat.csv", index=False)
