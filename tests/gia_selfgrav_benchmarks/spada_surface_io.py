"""Firedrake surface time series shared by the writer and offline converter.

One CheckpointFile stores the mesh and indexed Functions. A small manifest
records physical times and complete, active frames, independently of the
restart checkpoints. The converter needs only this file and Firedrake.
"""

import json
import os
from pathlib import Path

import firedrake as fd
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

GROUP = "spada_surface"
VERSION = 1


def _write_json(group, name, value):
    group.create_dataset(name, data=np.frombuffer(json.dumps(value).encode(), dtype=np.uint8))


def read_metadata(checkpoint):
    """Read our manifest without depending on Firedrake's internal HDF5 paths."""
    group = checkpoint.h5pyfile.get(GROUP)
    try:
        if group is None or group.attrs.get("version") != VERSION:
            raise ValueError("Not a supported Spada surface HDF5 file")
        return json.loads(group["metadata"][()].tobytes().decode())
    finally:
        del group


def read_frames(checkpoint):
    """Return complete, active frames, ordered by physical time."""
    frames = checkpoint.h5pyfile[GROUP]["frames"]
    return sorted(({"index": int(index), "t_kyr": float(frame.attrs["t_kyr"])}
                   for index, frame in frames.items()
                   if frame.attrs.get("complete", False) and frame.attrs.get("active", False)),
                  key=lambda frame: frame["t_kyr"])


def _matching_nodes(source, target, degree):
    """Match CG node identities when a volume-checkpoint reload reorders them.

    Both meshes describe the same physical surface, on the original rank
    count/partition. This is a permutation of nodal values, not evaluation
    or interpolation of a field at searched points. Source halos cover
    nodes that have changed owner. Only used when resuming an archive.
    """
    def coordinates(mesh):
        return fd.Function(fd.VectorFunctionSpace(mesh, "CG", degree)).interpolate(
            fd.SpatialCoordinate(mesh))

    source_points = coordinates(source).dat.data_ro_with_halos
    target_points = coordinates(target).dat.data_ro
    radius = source.comm.allreduce(float(np.max(np.abs(source_points), initial=0.)), op=MPI.MAX)
    tolerance = max(1., radius) * 1e-11
    distances, indices = cKDTree(source_points).query(target_points)
    valid = np.all(distances < tolerance) and len(np.unique(indices)) == len(indices)
    if not source.comm.allreduce(bool(valid), op=MPI.LAND):
        raise ValueError("Restart surface mesh nodes do not match the saved surface; "
                         "use a separate --output_path")
    return indices


def _invalidate_after(group, restart_time):
    for frame in group["frames"].values():
        if frame.attrs.get("t_kyr", float("inf")) > restart_time:
            frame.attrs["active"] = False


class SurfaceHDF5Series:
    """Append surface fields; close the collective file after every frame."""

    def __init__(self, path, mesh, metadata, *, restart_time=None):
        self.path, self.mesh = str(path), mesh
        self.metadata = metadata
        self._copies = {}
        self._permutations = {}
        self._stored_mesh = mesh
        exists = mesh.comm.bcast(Path(path).exists() if mesh.comm.rank == 0 else None, root=0)
        appending = restart_time is not None and exists
        if exists and not appending:
            if mesh.comm.rank == 0:
                os.replace(path, Path(path).with_suffix(".previous.h5"))
            mesh.comm.barrier()
        with fd.CheckpointFile(self.path, "a" if appending else "w", comm=mesh.comm) as checkpoint:
            if appending:
                saved = read_metadata(checkpoint)
                ranks = checkpoint.h5pyfile[GROUP].attrs["writer_ranks"]
                if saved != metadata or ranks != mesh.comm.size:
                    raise ValueError("Surface output mesh/layout or fields differ from the saved file; "
                                     "use a separate --output_path for this restart")
                self._stored_mesh = checkpoint.load_mesh(saved["mesh_name"])
                self._permutations = {degree: _matching_nodes(mesh, self._stored_mesh, degree)
                                      for degree in (1, 2, 3)}
                # An interrupted job may have produced epochs after the restored
                # checkpoint. Keep their storage indices reserved, but exclude
                # those obsolete frames from conversion.
                _invalidate_after(checkpoint.h5pyfile[GROUP], restart_time)
            else:
                checkpoint.save_mesh(mesh)
                group = checkpoint.h5pyfile.create_group(GROUP)
                group.attrs["version"] = VERSION
                group.attrs["writer_ranks"] = mesh.comm.size
                _write_json(group, "metadata", metadata)
                group.create_group("frames")
                del group
            self.next_index = max(map(int, checkpoint.h5pyfile[GROUP]["frames"]), default=-1) + 1
            frames = read_frames(checkpoint)
            self.last_time = frames[-1]["t_kyr"] if frames else -float("inf")

    def _stored_field(self, field):
        if not self._permutations:
            return field
        name = field.name()
        if name not in self._copies:
            space = fd.FunctionSpace(self._stored_mesh, field.function_space().ufl_element())
            self._copies[name] = fd.Function(space, name=name)
        target = self._copies[name]
        degree = field.function_space().ufl_element().degree()
        target.dat.data_wo[:] = field.dat.data_ro_with_halos[self._permutations[degree]]
        return target

    def write(self, fields, t_kyr, rate_info):
        if not np.isfinite(t_kyr) or t_kyr <= self.last_time:
            raise ValueError("Surface output times must be finite and strictly increasing")
        index = self.next_index
        with fd.CheckpointFile(self.path, "a", comm=self.mesh.comm) as checkpoint:
            frame = checkpoint.h5pyfile[GROUP]["frames"].create_group(str(index))
            try:
                frame.attrs.update(t_kyr=float(t_kyr), complete=False, active=True)
                info = {"t_kyr": float(t_kyr),
                        **{key: float(rate_info[key]) for key in
                           ("rate_dt_yr", "rate_interval_start_kyr", "rate_interval_end_kyr")}}
                for field in fields:
                    checkpoint.save_function(self._stored_field(field), idx=index, timestepping_info=info)
                # Publish only after every field is saved. Partial frames are never
                # converted, and a restart allocates a fresh index above them.
                checkpoint.h5pyfile.flush()
                frame.attrs["complete"] = True
            finally:
                del frame
        self.next_index += 1
        self.last_time = float(t_kyr)
