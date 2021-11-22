#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=as # Assamese

train_set=train_"$(echo "${lang}" | tr - _)"
train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set="${train_dev} test_$(echo ${lang} | tr - _)"

asr_config=conf/tuning/train_asr_transformer_pretrain.yaml
inference_config=conf/decode_asr.yaml

nbpe=100

./asr.sh \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --use_lm false \
    --token_type bpe \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" "$@"
