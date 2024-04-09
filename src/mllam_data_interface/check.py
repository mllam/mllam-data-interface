from pathlib import Path

import xarray as xr
import yaml
import importlib

SPECS_ROOT_PATH = Path(__file__).parent.parent.parent / "specs"


def load_spec(spec_identifier: str):
    """
    Load dataset specification from the given identifier.
    The specification is a string with format `{spec_name}:{spec_version}`.

    Parameters
    ----------
    spec_identifier : str
        The dataset specification identifier, e.g. `neural_lam:v0.1.0`

    Returns
    -------
    dict
        The dataset specification.
    """
    if ":" not in spec_identifier:
        raise ValueError(
            "The spec_identifier must be in the format {spec_name}:{spec_version}"
        )

    spec_name, spec_version = spec_identifier.split(":")

    # fp = SPECS_ROOT_PATH / spec_name / f"{spec_version}.yaml"
    fp = importlib.resources.files(__package__) / "specs" / spec_name / f"{spec_version}.yaml"
    if not fp.exists():
        raise ValueError(f"Spec file {fp} for {spec_identifier} does not exist.")

    with open(fp, "r") as f:
        spec = yaml.safe_load(f)
    
    return spec


def check_dataset(ds: xr.Dataset, spec_identifier: str):
    """
    Check that xr.Dataset `ds` satisfies the specification `spec_identifier` by
    container the necessary variables, dimensions and attributes.
    """

    spec = load_spec(spec_identifier=spec_identifier)

    required_variables = spec.get("variables", {})

    for var_name, var_parts in required_variables.items():
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable {var_name} is missing from the dataset.")

        dims = var_parts.get("dims", [])

        if not all(dim in ds[var_name].dims for dim in dims):
            raise ValueError(
                f"Variable {var_name} does not have the required dimensions {dims}."
            )

    required_attributes = spec.get("attributes", [])

    for attr in required_attributes:
        if attr not in ds.attrs:
            raise ValueError(f"Attribute {attr} is missing from the dataset.")


if __name__ == "__main__":
    import argparse

    from loguru import logger

    argparser = argparse.ArgumentParser(description="MLLAM Data Interface")
    argparser.add_argument("zarr_filepath", help="Path to the Zarr file to check")
    argparser.add_argument(
        "spec_identifier",
        help="The identifier of the spec to check, e.g. `neural_lam:v0.1.0`",
    )
    args = argparser.parse_args()

    logger.info(f"Opening Zarr file at {args.zarr_filepath}")
    ds = xr.open_zarr(args.zarr_filepath)
    spec_identifier = args.spec_identifier
    try:
        check_dataset(ds=ds, spec_identifier=spec_identifier)
        logger.info("Dataset matches the spec!")
    except Exception as ex:
        logger.exception(ex)
