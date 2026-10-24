"""Spada adapter for dsroberts/firedrake_h5_to_vtk (used unchanged).

Upstream handles mesh/field discovery, checkpoint loading and VTK export.
This adapter selects complete, active frames and restores physical PVD times.
Install upstream on PYTHONPATH, set FIREDRAKE_H5_TO_VTK to its script, or
pass --converter /path/to/firedrake_h5_to_vtk.py. See README.md.
"""

import argparse
import importlib
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import firedrake as fd
from spada_surface_io import read_frames, read_metadata


def load_converter(path=None):
    path = path or os.environ.get("FIREDRAKE_H5_TO_VTK")
    if path:
        spec = importlib.util.spec_from_file_location("firedrake_h5_to_vtk", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import upstream converter: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module("firedrake_h5_to_vtk")
    except ModuleNotFoundError as error:
        raise ValueError("Install dsroberts/firedrake_h5_to_vtk on PYTHONPATH or pass "
                         "--converter /path/to/firedrake_h5_to_vtk.py") from error


def convert_surface(infile, outfile=None, *, functions="all", timesteps="all", list_only=False,
                    converter=None):
    """Collectively delegate conversion to upstream, retaining Spada frame semantics."""
    if not list_only and (outfile is None or Path(outfile).suffix != ".pvd"):
        raise ValueError("Output file must end in .pvd")
    upstream = load_converter(converter)
    with fd.CheckpointFile(str(infile), "r") as checkpoint:
        metadata, frames = read_metadata(checkpoint), read_frames(checkpoint)
    if not frames:
        raise ValueError("No complete surface frames selected")
    available = metadata["fields"]
    if functions != "all" and set(functions.split(",")) - set(available):
        raise ValueError(f"Unknown fields in {functions}; available: {available}")
    function_filter = upstream.FunctionFilter(functions)
    names = [name for name in available if function_filter(name)]
    available_indices = [frame["index"] for frame in frames]
    indices = upstream.TimestepFilter(timesteps)(available_indices)
    if timesteps not in ("all", "last") and set(map(int, timesteps.split(","))) - set(available_indices):
        raise ValueError(f"Unknown frames in {timesteps}; available: {available_indices}")
    frames = [frame for frame in frames if frame["index"] in indices]
    if not frames:
        raise ValueError("No complete surface frames selected")
    if list_only:
        if fd.COMM_WORLD.rank == 0:
            print("Fields: " + ", ".join(names))
            for frame in frames:
                print(f"Frame {frame['index']}: {frame['t_kyr']:g} kyr")
        return
    arguments = [str(Path(upstream.__file__)), "-i", str(infile), "-o", str(outfile),
                 "-f", ",".join(names), "-t", ",".join(map(str, indices))]
    previous_argv = sys.argv
    try:
        sys.argv = arguments
        upstream.main()
    finally:
        sys.argv = previous_argv
    # Upstream currently writes frame numbers. Relabel the collection only;
    # all mesh/field data and VTU files are exactly those upstream produced.
    error = None
    if fd.COMM_WORLD.rank == 0:
        try:
            tree = ET.parse(outfile)
            datasets = tree.findall(".//DataSet")
            ordered = sorted(frames, key=lambda frame: frame["index"])
            if len(datasets) != len(ordered):
                raise ValueError("Upstream output frame count differs from selected Spada frames")
            for dataset, frame in zip(datasets, ordered):
                dataset.set("timestep", str(frame["t_kyr"]))
            temporary = Path(str(outfile) + ".tmp")
            tree.write(temporary, encoding="utf-8", xml_declaration=True)
            temporary.replace(outfile)
        except (OSError, ValueError, ET.ParseError) as exc:
            error = str(exc)
    error = fd.COMM_WORLD.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--infile", required=True, help="Spada surface .h5 file")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("-o", "--outfile", help="output .pvd file")
    output.add_argument("-l", "--list", action="store_true", help="list saved fields and physical times")
    parser.add_argument("-f", "--functions", default="all", help="all, or comma-separated field names")
    parser.add_argument("-t", "--timesteps", default="all",
                        help="all, last, or comma-separated frame indices shown by --list")
    parser.add_argument("--converter", help="path to upstream firedrake_h5_to_vtk.py (used unchanged)")
    args = parser.parse_args(argv)
    try:
        convert_surface(args.infile, args.outfile, functions=args.functions,
                        timesteps=args.timesteps, list_only=args.list, converter=args.converter)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
