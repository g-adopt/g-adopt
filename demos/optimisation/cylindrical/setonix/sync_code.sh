#!/bin/bash
# Copy one commit of this repository to a directory on Setonix, and prove the copy.
#
# Usage (on reunion, from anywhere inside the repository):
#     sync_code.sh <commit> <Setonix directory>
#
# The same procedure as demos/optimisation/box/prior_campaign/sync_code.sh, with two
# differences that the cylindrical comparison needs:
#   - the md5 proof covers every directory that the comparison imports from:
#     gadopt/, demos/optimisation/box/, demos/optimisation/cylindrical/,
#     tests/adjoint_2d_cylindrical/ and tests/hessian/;
#   - tests/adjoint_2d_cylindrical/Checkpoint230.h5 is stored in git LFS, so
#     `git archive` writes the LFS pointer and not the file. The real file is
#     copied from the working tree after the unpack, and its md5 is compared.
#
# The tracked tree of <commit> goes over as a `git archive` stream, so the copy holds
# exactly the committed files and no working-tree change. Setonix accepts the ssh key
# of phoenix only, so every command goes through phoenix. If the directory exists, it
# becomes <dir>.previous (one generation is kept). A CODE_STATE file records the
# commit.
set -euo pipefail

COMMIT=$(git rev-parse --verify "$1^{commit}")
DEST=$2
SHORT=$(git rev-parse --short=8 "$COMMIT")
SUBJECT=$(git log -1 --format=%s "$COMMIT")
TOP=$(git rev-parse --show-toplevel)
# The directories whose every tracked file is compared after the copy.
CHECKED="gadopt demos/optimisation/box demos/optimisation/cylindrical tests/adjoint_2d_cylindrical tests/hessian"
# The LFS file that the archive does not hold.
LFS_FILE=tests/adjoint_2d_cylindrical/Checkpoint230.h5

remote() {
    # Run a bash script from standard input on Setonix, through phoenix.
    ssh phoenix "ssh setonix 'bash -s'"
}

md5_stdin() {
    # The md5 of standard input: `md5 -q` on macOS, `md5sum` elsewhere.
    if command -v md5 > /dev/null; then md5 -q; else md5sum | cut -d' ' -f1; fi
}

echo "### unpack $SHORT into $DEST.incoming"
printf 'rm -rf %q.incoming && mkdir -p %q.incoming\n' "$DEST" "$DEST" | remote
git archive --format=tar "$COMMIT" \
    | ssh phoenix "ssh setonix 'tar -x -C $DEST.incoming'"

echo "### copy the LFS file $LFS_FILE"
# Through phoenix as a stream, so that no copy is left on phoenix.
ssh phoenix "ssh setonix 'cat > $DEST.incoming/$LFS_FILE'" < "$TOP/$LFS_FILE"

echo "### swap the copies"
remote <<REMOTE
set -euo pipefail
cd "\$(dirname "$DEST")"
NAME=\$(basename "$DEST")
if [ -d "\$NAME" ]; then
    rm -rf "\$NAME.previous"
    mv "\$NAME" "\$NAME.previous"
fi
mv "\$NAME.incoming" "\$NAME"
cat > "\$NAME/CODE_STATE" <<STATE
Source: branch sghelichkhani/hessian, commit $COMMIT
("$SUBJECT"), copied with git archive on $(date -u +%Y-%m-%d) by
demos/optimisation/cylindrical/setonix/sync_code.sh, plus the LFS file $LFS_FILE.
The tree holds the committed files only; untracked files are not part of it.
STATE
REMOTE

echo "### md5 of every tracked file under: $CHECKED"
export LC_ALL=C
LOCAL=$(mktemp); THERE=$(mktemp); SCRIPT=$(mktemp)
trap 'rm -f "$LOCAL" "$THERE" "$SCRIPT"' EXIT
git ls-tree -r --name-only "$COMMIT" -- $CHECKED | sort | while read -r path; do
    if [ "$path" = "$LFS_FILE" ]; then
        # The working-tree file, which is what was sent, not the pointer blob.
        printf '%s  %s\n' "$(md5_stdin < "$TOP/$path")" "$path"
    else
        printf '%s  %s\n' "$(git cat-file blob "$COMMIT:$path" | md5_stdin)" "$path"
    fi
done > "$LOCAL"
printf 'cd %q\n' "$DEST" > "$SCRIPT"
cut -c35- "$LOCAL" | while read -r path; do printf 'md5sum %q\n' "$path"; done >> "$SCRIPT"
remote < "$SCRIPT" | sort -k2 > "$THERE"
if diff -q "$LOCAL" "$THERE" > /dev/null; then
    echo "all $(wc -l < "$LOCAL" | tr -d ' ') tracked files equal"
else
    echo "files that differ:"; diff "$LOCAL" "$THERE" || true
    exit 1
fi
# Files on Setonix in the checked directories that the commit does not track. A stale
# module there could be imported, so the script names every one.
printf 'cd %q && find %s -type f -not -path "*/__pycache__/*"\n' "$DEST" "$CHECKED" \
    | remote | sort > "$THERE"
cut -c35- "$LOCAL" | sort | comm -13 - "$THERE" | sed 's/^/untracked on Setonix: /'
