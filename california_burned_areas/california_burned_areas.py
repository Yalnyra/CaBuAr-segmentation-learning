# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List
import torch 
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import h5py
import numpy as np

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{cabuar,
  title={Ca{B}u{A}r: California {B}urned {A}reas dataset for delineation},
  author={Rege Cambrin, Daniele and Colomba, Luca and Garza, Paolo},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  doi={10.1109/MGRS.2023.3292467},
  year={2023} 
}
"""

# You can copy an official description
_DESCRIPTION = """\
CaBuAr dataset contains images from Sentinel-2 satellites taken before and after a wildfire. 
The ground truth masks are provided by the California Department of Forestry and Fire Protection and they are mapped on the images.
"""

_HOMEPAGE = "https://huggingface.co/datasets/DarthReca/california_burned_areas"

_LICENSE = "OPENRAIL"

_URLS = ["raw/patched/512x512.hdf5", "raw/patched/chabud_test.h5"]

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (modify as needed)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts numpy arrays to torch tensors
    # Add more transforms if necessary, e.g., normalization
    # transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load the CaBuAr dataset
# Replace 'ca_bu_ar_module' with the actual module name if it's a local/custom dataset
# If it's a dataset on Hugging Face Hub, use its identifier
dataset = datasets.load_dataset('ca_bu_ar_module', name='pre-post-fire', split='train')


class CaBuArConfig(datasets.BuilderConfig):
    """BuilderConfig for CaBuAr.

    Parameters
    ----------

    load_prefire: bool
        whether to load prefire data
    train_folds: List[int]
        list of folds to use for training
    validation_folds: List[int]
        list of folds to use for validation
    test_folds: List[int]
        list of folds to use for testing
    **kwargs
        keyword arguments forwarded to super.
    """

    def __init__(self, load_prefire: bool, **kwargs):
        super(CaBuArConfig, self).__init__(**kwargs)
        self.load_prefire = load_prefire


class CaBuAr(datasets.GeneratorBasedBuilder):
    """California Burned Areas dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        CaBuArConfig(
            name="post-fire",
            version=VERSION,
            description="Post-fire only version of the dataset",
            load_prefire=False,
        ),
        CaBuArConfig(
            name="pre-post-fire",
            version=VERSION,
            description="Pre-fire and post-fire version of the dataset",
            load_prefire=True,
        ),
    ]

    DEFAULT_CONFIG_NAME = "post-fire"
    BUILDER_CONFIG_CLASS = CaBuArConfig

    def _info(self):
        if self.config.name == "pre-post-fire":
            features = datasets.Features(
                {
                    "post_fire": datasets.Array3D((512, 512, 12), dtype="uint16"),
                    "pre_fire": datasets.Array3D((512, 512, 12), dtype="uint16"),
                    "mask": datasets.Array3D((512, 512, 1), dtype="uint16"),
                }
            )
        else:
            features = datasets.Features(
                {
                    "post_fire": datasets.Array3D((512, 512, 12), dtype="uint16"),
                    "mask": datasets.Array3D((512, 512, 1), dtype="uint16"),
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        h5_files = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=fold,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "fold": fold,
                    "load_prefire": self.config.load_prefire,
                    "filepath": h5_files[file_index],
                },
            )
            for fold, file_index in zip(list(range(0, 5)) + ["chabud"], [0] * 5 + [1])
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, fold: int, load_prefire: bool, filepath):
        with h5py.File(filepath, "r") as f:
            for uuid, values in f.items():
                if values.attrs["fold"] != fold:
                    continue
                if load_prefire and "pre_fire" not in values:
                    continue
                sample = {
                    "post_fire": values["post_fire"][...],
                    "mask": values["mask"][...],
                }
                if load_prefire:
                    sample["pre_fire"] = values["pre_fire"][...]
                yield uuid, sample


# Define a custom PyTorch Dataset to apply transformations
class CaBuArPyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, load_prefire=True):
        """
        Args:
            hf_dataset: Hugging Face dataset object
            transform: Optional transform to be applied on a sample
            load_prefire: Boolean indicating whether to load pre_fire data
        """
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.load_prefire = load_prefire

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        post_fire = sample['post_fire']  # Shape: (512, 512, 12)
        mask = sample['mask']            # Shape: (512, 512, 1)
        
        if self.load_prefire:
            pre_fire = sample.get('pre_fire', np.zeros_like(post_fire))
        else:
            pre_fire = None

        # Apply transformations
        if self.transform:
            post_fire = self.transform(post_fire)
            mask = self.transform(mask)
            if pre_fire is not None:
                pre_fire = self.transform(pre_fire)
        
        # Move tensors to device
        post_fire = post_fire.to(device)
        mask = mask.to(device)
        if pre_fire is not None:
            pre_fire = pre_fire.to(device)
        
        # Prepare the output dictionary
        if self.load_prefire:
            return {
                'post_fire': post_fire,    # Tensor: [12, 512, 512]
                'pre_fire': pre_fire,      # Tensor: [12, 512, 512]
                'mask': mask.squeeze(0)    # Tensor: [512, 512]
            }
        else:
            return {
                'post_fire': post_fire,    # Tensor: [12, 512, 512]
                'mask': mask.squeeze(0)    # Tensor: [512, 512]
            }

# Instantiate the custom dataset
pytorch_dataset = CaBuArPyTorchDataset(
    hf_dataset=dataset,
    transform=transform,
    load_prefire=True  # Set to False if you want to load only post_fire data
)

# Create a DataLoader
dataloader = DataLoader(
    pytorch_dataset,
    batch_size=16,       # Adjust batch size as needed
    shuffle=True,        # Shuffle for training
    num_workers=4,       # Number of subprocesses for data loading
    pin_memory=True      # Speed up transfer to GPU
)

# Example: Iterate through the DataLoader
for batch_idx, batch in enumerate(dataloader):
    post_fire = batch['post_fire']      # Shape: [batch_size, 12, 512, 512]
    mask = batch['mask']                # Shape: [batch_size, 512, 512]
    pre_fire = batch.get('pre_fire')    # Shape: [batch_size, 12, 512, 512] or None

    # Now you can pass `post_fire`, `pre_fire`, and `mask` to your model
    # Example:
    # outputs = model(post_fire, pre_fire)
    # loss = criterion(outputs, mask)
    
    # For demonstration, we'll just print the batch shapes
    print(f"Batch {batch_idx}:")
    print(f"  post_fire shape: {post_fire.shape}")
    if pre_fire is not None:
        print(f"  pre_fire shape: {pre_fire.shape}")
    print(f"  mask shape: {mask.shape}")
    
    # Break after first batch for demonstration
    if batch_idx == 0:
        break