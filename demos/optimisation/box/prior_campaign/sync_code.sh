#!/bin/bash
# Copy one commit of this repository to a directory on Setonix, and prove the copy.
#
# Usage (on reunion, from anywhere inside the repository):
#     sync_code.sh <commit> <Setonix directory>
#
# The tracked tree of <commit> goes over as a `git archive` stream, so the copy holds
# exactly the committed files and no working-tree change. Setonix accepts the ssh key
# of phoenix only, so every command goes through phoenix.
#
# If the directory exists, the new tree replaces it: the archive is unpacked into
# <dir>.incoming, the untracked data files listed in KEEP move over from the old copy,
# the old copy becomes <dir>.previous (one generation is kept), and <dir>.incoming
# becomes <dir>. A CODE_STATE file records the commit.
#
# The proof: the md5 of every tracked file under gadopt/ and demos/optimisation/box/
# at <commit>, computed from the git objects on reunion and from the files on Setonix.
# The script prints the number of files that differ and exits 1 if any does.
set -euo pipefail

COMMIT=$(git rev-parse --verify "$1^{commit}")
DEST=$2
SHORT=$(git rev-parse --short=8 "$COMMIT")
SUBJECT=$(git log -1 --format=%s "$COMMIT")
# Untracked data files that the tests need and the repository does not hold.
KEEP="tests/adjoint/adjoint-demo-checkpoint-state.h5"
# The directories whose every tracked file is compared after the copy.
CHECKED="gadopt demos/optimisation/box"

remote() {
    # Run a bash script from standard input on Setonix, through phoenix.
    ssh phoenix "ssh setonix 'bash -s'"
}

echo "### unpack $SHORT into $DEST.incoming"
printf 'rm -rf %q.incoming && mkdir -p %q.incoming\n' "$DEST" "$DEST" | remote
git archive --format=tar "$COMMIT" \
    | ssh phoenix "ssh setonix 'tar -x -C $DEST.incoming'"

echo "### swap the copies"
remote <<REMOTE
set -euo pipefail
cd "\$(dirname "$DEST")"
NAME=\$(basename "$DEST")
if [ -d "\$NAME" ]; then
    for f in $KEEP; do
        if [ -e "\$NAME/\$f" ]; then
            mkdir -p "\$NAME.incoming/\$(dirname "\$f")"
            mv "\$NAME/\$f" "\$NAME.incoming/\$f"
        fi
    done
    rm -rf "\$NAME.previous"
    mv "\$NAME" "\$NAME.previous"
fi
mv "\$NAME.incoming" "\$NAME"
cat > "\$NAME/CODE_STATE" <<STATE
Source: branch sghelichkhani/hessian, commit $COMMIT
("$SUBJECT"), copied with git archive on $(date -u +%Y-%m-%d) by sync_code.sh.
The tree holds the committed files only; untracked files are not part of it.
STATE
REMOTE

echo "### md5 of every tracked file under: $CHECKED"
export LC_ALL=C
LOCAL=$(mktemp); THERE=$(mktemp); SCRIPT=$(mktemp)
trap 'rm -f "$LOCAL" "$THERE" "$SCRIPT"' EXIT
md5_stdin() {
    # The md5 of standard input: `md5 -q` on macOS, `md5sum` elsewhere.
    if command -v md5 > /dev/null; then md5 -q; else md5sum | cut -d' ' -f1; fi
}
git ls-tree -r --name-only "$COMMIT" -- $CHECKED | sort | while read -r path; do
    printf '%s  %s\n' "$(git cat-file blob "$COMMIT:$path" | md5_stdin)" "$path"
done > "$LOCAL"
# The same list of paths, hashed on Setonix by md5sum, which prints the same format.
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
