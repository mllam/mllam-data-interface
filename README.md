# mllam-data-interface

**Simple interfaces for describing weather machine learning datasets**

The purpose of `mllam_data_interface` is to provide a simple way to:

- describe a specification for what variables and attributes a specific
  `pytorch.Dataset` expects a `zarr`-based training dataset to contain
- check a `zarr` training-dataset against a specification
- provide storage for a collection of dataset specifications (these are in
  [specs/](src/mllam_data_interface/specs/) with path format
  `specs/{spec_name}/{spec_version}.yaml`)

# Usage

The intention with this package is that you can use it both when a) creating a
dataset and b) loading a dataset to check that the dataset contains what it
needs to. For convenience you can also [run it from the command line](#usage-from-command-line).

## Usage when loading a dataset

E.g. when loading a dataset:

```python
import mllam_data_interface as mdi
import xarray as xr

class MyWeatherDataset(pytorch.Dataset):
    mllam_spec = "neural_lam:v0.1.0"

    def __init__(self, dataset_path):
        self.ds = xr.open_zarr(self.dataset_path)
        mdi.check_dataset(ds=self.ds, spec_identifier=self.mllam_spec)
```

## Usage from command line

To check that a training dataset matches a spec you can use
`mllam_data_interface` from the command line:

```bash
python -m mllam_data_interface.check my_dataset.zarr neural_lam:v0.1.0
```

Which will output something like (if the dataset matches the spec):

```bash
python -m mllam_data_interface.check ../mllam-data-prep/example.danra.zarr neural_lam:v0.1.0
2024-04-09 11:04:21.943 | INFO     | __main__:<module>:72 - Opening Zarr file at example.danra.zarr
2024-04-09 11:04:22.023 | INFO     | __main__:<module>:76 - Dataset matches the spec!
```

Or (if the dataset doesn't match the spec):

```bash
2024-04-09 11:06:11.439 | INFO     | __main__:<module>:72 - Opening Zarr file at example.danra.incomplete.zarr
2024-04-09 11:06:11.518 | ERROR    | __main__:<module>:79 - Variable static is missing from the dataset.
```

## Specification format

All specifications are given as `yaml`-files. Currently, the specifications allow you to specify:

1. Which variables the dataset must contain, and which dimensions each should have (the order of the dimensions is also checked)
2. Which attributes the dataset must have

For example, the spec `neural_lam:v0.1.0` specifies that the dataset must contain the variables `static`, `state` and `forcing` variables could be written as:

```yaml
variables:
  static:
    dims: [grid_index, feature]
  state:
    dims: [time, grid_index, state_feature]
  forcing:
    dims: [time, grid_index, forcing_feature]
attributes: [version]
```
