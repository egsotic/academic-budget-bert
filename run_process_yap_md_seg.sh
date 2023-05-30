#!/bin/bash

# setup
cd /home/nlp/egsotic/repo/academic-budget-bert
source ./venv-24h-bert-link/bin/activate
cd dataset

data_path=$1
raw_tokens_path=${data_path/\/shards\//\/yap\/raw\/}
ma_path=${data_path/\/shards\//\/yap\/ma\/}
md_path=${data_path/\/shards\//\/yap\/md\/}
output_path=${data_path/\/shards\//\/shards_yap_md_seg\/}

echo "data path: ${data_path}"

# lines -> raw format
echo "lines -> raw format"
if [ ! -f "${raw_tokens_path}" ]; then
    echo "output file: ${raw_tokens_path}"
    python yap_lines_to_raw.py --input_path "${data_path}" --output_path "${raw_tokens_path}"
else
    echo "File already exists"
fi

# ma
echo " -> ma"
if { [ ! -f "${ma_path}" ] || [ ! -s "${ma_path}" ]; } && [ ! -f "${md_path}" ]; then
    echo "output file: ${ma_path}"
    /home/nlp/egsotic/go/src/yap/yap hebma -addnnpnofeats -raw "${raw_tokens_path}" -out "${ma_path}" > /dev/null 2>&1
else
    echo "File already exists"
fi

# md
echo " -> md"
if [ ! -f "${md_path}" ]; then
    echo "output file: ${md_path}"
    /home/nlp/egsotic/go/src/yap/yap md -stripnnpfeats -stream -in "${ma_path}" -om "${md_path}" > /dev/null 2>&1
else
    echo "File already exists"
fi

# rm ma
if [ -f "${ma_path}" ]; then
    rm "${ma_path}"
fi

# output md seg
echo " -> seg"
if [ ! -f "${output_path}" ]; then
echo "output file: ${output_path}"
    python yap_md_to_seg.py --input_path "${md_path}" --output_path "${output_path}"
else
    echo "File already exists"
fi

# TODO: rm md ?
