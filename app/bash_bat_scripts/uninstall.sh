#!/usr/bin/env bash
PREFIX="BASE_PATH"
echo "Uninstalling StarDist2D from $PREFIX"
if [ -f "$PREFIX/pre_uninstall.sh" ]; then
    # best effort: shortcut/registry cleanup must not stop the removal, but a failure must be visible
    if ! bash "$PREFIX/pre_uninstall.sh"; then
        echo "WARNING: pre-uninstall cleanup failed (see the messages above); shortcuts or the tools registry entry may remain." >&2
    fi
fi
if ! rm -rf -- "$PREFIX"; then
    echo "ERROR: could not remove $PREFIX; StarDist2D is only partly uninstalled. Close anything using it and delete that folder manually." >&2
    exit 1
fi

echo "StarDist2D removed."

if [ -t 0 ]; then
    echo
    read -rp "Press Enter to close the installer..." _
fi