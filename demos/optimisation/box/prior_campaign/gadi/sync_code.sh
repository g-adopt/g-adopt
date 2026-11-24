#!/bin/bash
# Copy one commit of this repository to Gadi, and prove the copy.
#
# Usage (on reunion, from anywhere inside the repository):
#     sync_code.sh <commit> [<Gadi directory>]
#
# The default directory is /scratch/nm03/sg8812/hessian/gauss-newton/code-<short>/.
# The tracked tree of <commit> goes over as a `git archive` stream, so the copy holds
# exactly the committed files. git LFS files arrive as pointers; the box code reads
# none of them. A CODE_STATE file records the commit. An existing directory is never
# overwritten: a code copy that a job may be reading must not change.
#
# The proof: the md5 of every tracked file under gadopt/ and demos/optimisation/box/
# at <commit>, from the git objects on reunion and from the files on Gadi. The script
# exits 1 if any file differs. Every remote command starts with `source /etc/profile`,
# without which Gadi does not find its commands.
set -euo pipefail
COMMIT=$(git rev-parse --verify "$1^{commit}")
SHORT=$(git rev-parse --short=8 "$COMMIT")
DEST=${2:-/scratch/nm03/sg8812/hessian/gauss-newton/code-$SHORT}
SUBJECT=$(git log -1 --format=%s "$COMMIT")
CHECKED="gadopt demos/optimisation/box"

remote() {
    # Run a bash script from standard input on Gadi.
    ssh gadi 'source /etc/profile && bash -s'
}

md5_stdin() {
    # The md5 of standard input: `md5 -q` on macOS, `md5sum` elsewhere.
    if command -v md5 > /dev/null; then md5 -q; else md5sum | cut -d' ' -f1; fi
}

if printf 'test -e %q\n' "$DEST" | remote; then
    echo "$DEST exists on Gadi; choose another directory" >&2
    exit 1
fi
echo "### unpack $SHORT into $DEST"
printf 'mkdir -p %q\n' "$DEST" | remote
git archive --format=tar "$COMMIT" | ssh gadi "source /etc/profile && tar -x -C $DEST"
remote <<REMOTE
cat > "$DEST/CODE_STATE" <<STATE
Source: branch sghelichkhani/hessian, commit $COMMIT
("$SUBJECT"), copied with git archive on $(date -u +%Y-%m-%d) by
demos/optimisation/box/prior_campaign/gadi/sync_code.sh.
The tree holds the committed files only; untracked files are not part of it.
STATE
REMOTE

echo "### md5 of every tracked file under: $CHECKED"
export LC_ALL=C
LOCAL=$(mktemp); THERE=$(mktemp); SCRIPT=$(mktemp)
trap 'rm -f "$LOCAL" "$THERE" "$SCRIPT"' EXIT
git ls-tree -r --name-only "$COMMIT" -- $CHECKED | sort | while read -r path; do
    printf '%s  %s\n' "$(git cat-file blob "$COMMIT:$path" | md5_stdin)" "$path"
done > "$LOCAL"
printf 'cd %q\n' "$DEST" > "$SCRIPT"
cut -c35- "$LOCAL" | while read -r path; do printf 'md5sum %q\n' "$path"; done >> "$SCRIPT"
remote < "$SCRIPT" | sort -k2 > "$THERE"
if diff -q "$LOCAL" "$THERE" > /dev/null; then
    echo "all $(wc -l < "$LOCAL" | tr -d ' ') tracked files equal in $DEST"
else
    echo "files that differ:"; diff "$LOCAL" "$THERE" || true
    exit 1
fi
