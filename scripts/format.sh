#bin/bash

# check for working directory
WORKING_DIR=$(pwd)
if [ ! -d "$WORKING_DIR/src" ]; then
  echo "Error: You must run this script from the root of the project"
  exit 1
fi

# run black
echo "Running black..."
black .
echo "Running isort..."
isort .
echo "Running autoflake..."
autoflake --in-place --remove-all-unused-imports --recursive --verbose src tests
