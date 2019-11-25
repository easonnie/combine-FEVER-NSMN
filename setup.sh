#!/usr/bin/env bash

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=${PYTHONPATH}:${DIR_TMP}/src
export PYTHONPATH=${PYTHONPATH}:${DIR_TMP}/utest
export PYTHONPATH=${PYTHONPATH}:${DIR_TMP}/dep_packages
export PYTHONPATH=${PYTHONPATH}:${DIR_TMP}/dep_packages/DrQA

echo PYTHONPATH=${PYTHONPATH}