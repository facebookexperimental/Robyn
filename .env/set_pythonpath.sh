#!/bin/bash

# Determine the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the PYTHONPATH to the absolute path of the src directory
export PYTHONPATH="$SCRIPT_DIR/../python/src"

# Normalize the PYTHONPATH to remove any relative path components
PYTHONPATH="$(cd "$PYTHONPATH" && pwd)"

# Check if the script is running in GitHub Actions
if [ -n "$GITHUB_ENV" ]; then
    # Export the PYTHONPATH to the GitHub Actions environment
    echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV
else
    # Export the PYTHONPATH for the current shell session
    export PYTHONPATH
    echo "PYTHONPATH set to $PYTHONPATH"
fi