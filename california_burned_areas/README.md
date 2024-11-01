---
license: openrail
task_categories:
- image-segmentation
pretty_name: California Burned Areas
size_categories:
- n<1K
tags:
- climate
---
# California Burned Areas Dataset

**Working on adding more data**

## Dataset Description

- **Paper:** [Pre-Print](https://arxiv.org/abs/2401.11519) and [Version of Record](https://ieeexplore.ieee.org/document/10261881)

### Dataset Summary

This dataset contains images from Sentinel-2 satellites taken before and after a wildfire. 
The ground truth masks are provided by the California Department of Forestry and Fire Protection and they are mapped on the images.

### Supported Tasks

The dataset is designed to do binary semantic segmentation of burned vs unburned areas.

## Dataset Structure

We opted to use HDF5 to grant better portability and lower file size than GeoTIFF.

### Dataset opening

Using the dataset library, you download only the pre-patched raw version for simplicity.
```python
from dataset import load_dataset

# There are two available configurations, "post-fire" and "pre-post-fire."
dataset = load_dataset("DarthReca/california_burned_areas", name="post-fire")
```

The dataset was compressed using `h5py` and BZip2 from `hdf5plugin`. **WARNING: `hdf5plugin` is necessary to extract data**.

### Data Instances

Each matrix has a shape of 5490x5490xC, where C is 12 for pre-fire and post-fire images, while it is 0 for binary masks.
Pre-patched version is provided with matrices of size 512x512xC, too. In this case, only mask with at least one positive pixel is present.

You can find two versions of the dataset: _raw_ (without any transformation) and _normalized_ (with data normalized in the range 0-255). 
Our suggestion is to use the _raw_ version to have the possibility to apply any wanted pre-processing step.

### Data Fields

In each standard HDF5 file, you can find post-fire, pre-fire images, and binary masks. The file is structured in this way:

```bash
├── foldn
│   ├── uid0
│   │   ├── pre_fire
│   │   ├── post_fire
│   │   ├── mask 
│   ├── uid1
│       ├── post_fire
│       ├── mask
│  
├── foldm
    ├── uid2
    │   ├── post_fire
    │   ├── mask 
    ├── uid3
        ├── pre_fire
        ├── post_fire
        ├── mask
...
```

where `foldn` and `foldm` are fold names and `uidn` is a unique identifier for the wildfire.

For the pre-patched version, the structure is:
```bash
root
|
|-- uid0_x: {post_fire, pre_fire, mask}
|
|-- uid0_y: {post_fire, pre_fire, mask}
|
|-- uid1_x: {post_fire, mask}
|
...
```
the fold name is stored as an attribute.

### Data Splits

There are 5 random splits whose names are: 0, 1, 2, 3, and 4.

### Source Data

Data are collected directly from Copernicus Open Access Hub through the API. The band files are aggregated into one single matrix.

## Additional Information

### Licensing Information

This work is under OpenRAIL license.

### Citation Information

If you plan to use this dataset in your work please give the credit to Sentinel-2 mission and the California Department of Forestry and Fire Protection and cite using this BibTex:
```
@ARTICLE{cabuar,
  author={Cambrin, Daniele Rege and Colomba, Luca and Garza, Paolo},
  journal={IEEE Geoscience and Remote Sensing Magazine}, 
  title={CaBuAr: California burned areas dataset for delineation [Software and Data Sets]}, 
  year={2023},
  volume={11},
  number={3},
  pages={106-113},
  doi={10.1109/MGRS.2023.3292467}
}
```