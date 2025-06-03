#!/bin/bash

rm -rf dist
rm -rf build
rm -rf maxim_py.egg-info
export PATH=$PATH:.venv/bin
source .venv/bin/activate
python3 setup.py sdist bdist_wheel
twine upload dist/*