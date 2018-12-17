#!/usr/bin/env bash

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/.."; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"


# check if data exist.
if [[ ! -d ${DIR_TMP}/data ]]; then
    mkdir ${DIR_TMP}/data
fi

# download the data.
cd ${DIR_TMP}/data
if [[ ! -d fever ]]; then
    mkdir fever
fi

if [[ ! -d tokenized_fever ]]; then
    mkdir tokenized_fever
fi

cd ${DIR_TMP}/data/fever
if [[ ! -f shared_task_dev.jsonl ]]; then
    wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
fi

if [[ ! -f train.jsonl ]]; then
    wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
fi

if [[ ! -f shared_task_test.jsonl ]]; then
    wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl
fi

cd ${DIR_TMP}/data
if [[ ! -d wiki-pages ]]; then
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip "wiki-pages.zip" && rm "wiki-pages.zip" && rm "__MACOSX"
fi

# download the dependency
cd ${DIR_TMP}
if [[ ! -d dep_packages ]]; then
    mkdir dep_packages
    cd dep_packages
    wget "https://www.dropbox.com/s/74uc24un1eoqwch/dep_packages.zip?dl=0"
    mv "dep_packages.zip?dl=0" "dep_packages.zip"
    unzip "dep_packages.zip" && rm "dep_packages.zip" && rm "__MACOSX"
fi
