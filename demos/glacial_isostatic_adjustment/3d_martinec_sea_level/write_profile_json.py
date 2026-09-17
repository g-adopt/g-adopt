"""Write `martinec_profiles.json` from the converted Martinec reference data.

`martinec_benchmark.py` evaluates U, N and S along the two comparison
meridians of each case, at the colatitudes of the VEGA profile. Those
colatitudes live in the converted reference data of the gia-mip checkout,
`data/converted/martinec2018/martinec2018-<case>.nc`, group
`profiles/<profile>/ZM`. The driver must not read NetCDF on a compute node, so
this script copies the colatitudes into a small JSON file next to the driver
and the driver reads that.

The three grids differ per case and per profile, because the contributors
delivered different ones: 2048 points where the VEGA output is on its own
Gauss-Legendre grid, and 359 points where the archive holds a coarse copy (see
the correction `martinec2018-caseD-vega-profile-grid` of gia-mip).

Run it on a machine that has the gia-mip checkout and `netCDF4`:

    python write_profile_json.py --gia-mip ~/Workplace/gia-mip

The file it writes is tracked, so it only has to run again when the converted
data change.
"""

import argparse
import json
import os

import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: The cases the driver runs.
CASES = ("B", "C", "D")
#: The two comparison meridians of every case.
PROFILES = ("load", "basin")
#: The reference contributor. VEGA is the reference solution of every case of
#: Martinec et al. (2018) and ZM is its code in the archive.
CONTRIBUTOR = "ZM"


def read_case(path, letter):
    """The two profiles of one case, as plain Python data.

    Args:
      path: the converted NetCDF file of the case.
      letter: the case letter, for the error message.

    Returns:
      A dictionary keyed by profile name, each with `longitude_deg`,
      `time_kyr`, `quantities` and `colatitude_deg`.

    Raises:
      KeyError: if a profile or the reference contributor is absent.
    """
    out = {}
    with netCDF4.Dataset(path) as data:
        groups = data.groups["profiles"].groups
        for name in PROFILES:
            if name not in groups:
                raise KeyError(f"case {letter}: no profile {name!r} in {path}")
            if CONTRIBUTOR not in groups[name].groups:
                raise KeyError(f"case {letter}, profile {name}: no "
                               f"contributor {CONTRIBUTOR!r} in {path}")
            group = groups[name].groups[CONTRIBUTOR]
            colatitude = np.asarray(group.variables["colatitude"][:],
                                    dtype=float)
            out[name] = {
                "longitude_deg": float(group.longitude_deg),
                "time_kyr": float(group.time_kyr),
                "quantities": sorted(key for key in group.variables
                                     if key != "colatitude"),
                "colatitude_deg": [float(value) for value in colatitude]}
    return out


def main():
    """Write the JSON file next to this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gia-mip",
                        default=os.environ.get(
                            "GIA_MIP",
                            os.path.expanduser("~/Workplace/gia-mip")),
                        help="the gia-mip checkout")
    parser.add_argument("--output",
                        default=os.path.join(HERE, "martinec_profiles.json"),
                        help="the JSON file to write")
    args = parser.parse_args()

    source = os.path.join(args.gia_mip, "data", "converted", "martinec2018")
    document = {
        "$comment": (
            "Colatitudes of the VEGA (contributor ZM) comparison profiles of "
            "Martinec et al. (2018) cases B, C and D. Written by "
            "write_profile_json.py from the converted reference data of "
            "gia-mip; read by martinec_benchmark.py."),
        "source": ("gia-mip data/converted/martinec2018/martinec2018-"
                   "{case}.nc, group profiles/{profile}/" + CONTRIBUTOR),
        "contributor": CONTRIBUTOR,
        "cases": {letter: read_case(
            os.path.join(source, f"martinec2018-{letter}.nc"), letter)
            for letter in CASES}}

    with open(args.output, "w") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    for letter, case in document["cases"].items():
        for name, entry in case.items():
            print(f"case {letter}, profile {name}: "
                  f"{len(entry['colatitude_deg'])} colatitudes at longitude "
                  f"{entry['longitude_deg']:g}, t = {entry['time_kyr']:g} kyr")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
