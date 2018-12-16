#!/usr/bin/env bash

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/.."; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"


# check if data exist
if [[ ! -d ${DIR_TMP}/data ]]; then
    mkdir ${DIR_TMP}/data
fi

# FEVER dev: https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
# FEVER test: https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl
# FEVER train: https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
# WIKI pages: https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip

# dependency package: