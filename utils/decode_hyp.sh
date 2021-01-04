#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
help_message="Usage: $0 <data-dir> <dict>"

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json

json2trn.py ${dir}/data.json ${dic} --num-spkrs 1 --refs ${dir}/ref.trn --hyps ${dir}/hyp.trn

if ${remove_blank}; then
      sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ -n "${nlsyms}" ]; then
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
fi


if ${wer}; then
    if [ -n "$bpe" ]; then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    else
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    fi
fi

