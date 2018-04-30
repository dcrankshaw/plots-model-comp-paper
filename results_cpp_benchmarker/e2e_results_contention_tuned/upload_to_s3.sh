#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# Let the user start this script from anywhere in the filesystem.
cd $DIR/..

aws s3 sync $DIR s3://inferline-results-contention-tuned \
  --exclude upload_to_s3.sh --exclude download_from_s3.sh --exclude .gitignore
