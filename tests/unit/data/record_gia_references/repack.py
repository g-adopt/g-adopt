"""Copy HDF5 files with gzip compression in the HDF5 1.10 file format.

Every dataset with more than 64 entries is written with gzip (level 9) and
the shuffle filter; groups, datasets and attributes keep their names and
values. HDF5 1.10 and later read the result, and so does Firedrake's
`CheckpointFile`. The files of `record_checkpoints.py` shrink by a factor
of about four this way. Needs h5py only, not the earlier solver.

Usage:
    python repack.py RAW_DIR OUT_DIR
"""
import os
import sys
import h5py

src_dir, dst_dir = sys.argv[1:3]
for name in sorted(os.listdir(src_dir)):
    with h5py.File(f"{src_dir}/{name}", "r") as src, h5py.File(f"{dst_dir}/{name}", "w", libver=("v110", "v110")) as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def copy(group_src, group_dst):
            for k, obj in group_src.items():
                if isinstance(obj, h5py.Group):
                    g = group_dst.create_group(k)
                    for a, b in obj.attrs.items():
                        g.attrs[a] = b
                    copy(obj, g)
                else:
                    kw = (dict(compression="gzip", compression_opts=9, shuffle=True)
                          if obj.shape and obj.size > 64 else {})
                    d = group_dst.create_dataset(k, data=obj[()], **kw)
                    for a, b in obj.attrs.items():
                        d.attrs[a] = b
        copy(src, dst)
    print(name, os.path.getsize(f"{src_dir}/{name}"), os.path.getsize(f"{dst_dir}/{name}"))
