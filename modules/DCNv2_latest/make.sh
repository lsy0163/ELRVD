#!/usr/bin/env bash
# Build the DCNv2 extension into the *current* Python environment.
# Run with the ELRVD conda env activated (do not use sudo: it would
# build against the system Python instead of the conda env).
rm -f *.so
rm -rf build/
python setup.py build develop
