import importlib.resources

import numpy as np
import pytest
import xarray as xr

from mllam_data_interface.check import check_dataset, load_spec

SPEC_IDENTIFIER_FORMAT = "{spec_name}:{spec_version}"


def _find_all_specs():
    fps = list(
        (importlib.resources.files("mllam_data_interface") / "specs").rglob("*.yaml")
    )

    # parse spec name and version from each path
    spec_identifiers = []
    for fp in fps:
        spec_name = fp.parent.name
        spec_version = fp.stem
        spec_identifier = SPEC_IDENTIFIER_FORMAT.format(
            spec_name=spec_name, spec_version=spec_version
        )
        spec_identifiers.append(spec_identifier)

    return spec_identifiers


@pytest.mark.parametrize("spec_identifier", _find_all_specs())
def test_parsing(spec_identifier):
    spec = load_spec(spec_identifier=spec_identifier)

    # generate a fake dataset that matches the spec
    ds = xr.Dataset()
    for var_name, var_parts in spec.get("variables", {}).items():
        dims = var_parts.get("dims", [])
        shape = [
            3,
        ] * len(dims)
        ds[var_name] = xr.DataArray(np.random.rand(*shape), dims=dims)

    for attr in spec.get("attributes", []):
        ds.attrs[attr] = "some_value"

    check_dataset(ds=ds, spec_identifier=spec_identifier)

    # check that an exception is raised with an empty dataset that doesn't match the
    # spec
    with pytest.raises(ValueError):
        check_dataset(ds=xr.Dataset(), spec_identifier=spec_identifier)
