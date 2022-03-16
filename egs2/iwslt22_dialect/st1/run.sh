#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=ta
tgt_lang=en

train_set=train
train_dev=dev
test_set=test1

# st_config=conf/md_conformer_try1.yaml
# inference_config=conf/decode_st.yaml
st_config=conf/train_st_conformer.yaml
# inference_config=conf/decode_st_conformer.yaml
inference_config=conf/decode_md.yaml

src_nbpe=1000
tgt_nbpe=1000

# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal
# Note, it is best to keep tgt_case as tc to match IWSLT22 eval
src_case=tc.rm
tgt_case=tc

./st.sh \
    --stage 12 \
    --stop_stage 13 \
    --use_lm false \
    --use_asrlm false \
    --use_mt false \
    --use_asr_inference_text false \
    --token_joint false \
    --audio_format "flac.ark" \
    --nj 40 \
    --inference_nj 40 \
    --st_tag hop112_lr2e-3_mtl0.3_raw_bpe_tc1000_sp \
    --audio_format "flac.ark" \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --use_ensemble true \
    --additional_st_train_config "/ocean/projects/cis210027p/pyf98/espnet-iwslt/egs2/iwslt22_dialect/st1/exp/st_train_st_conformer_lr2e-3_warmup15k_wdecay5e-6_mtl0.3_asr0.3_asrinit_raw_bpe_tc1000_sp/config.yaml" \
    --additional_st_model_file "/ocean/projects/cis210027p/pyf98/espnet-iwslt/egs2/iwslt22_dialect/st1/exp/st_train_st_conformer_lr2e-3_warmup15k_wdecay5e-6_mtl0.3_asr0.3_asrinit_raw_bpe_tc1000_sp/valid.acc.ave_10best.pth" \
    --additional_st_train_config "/ocean/projects/cis210027p/byan/espnet-md/egs2/iwslt22_dialect/st1/exp/st_hop112_md_ag4_raw_bpe_tc1000_sp/config.yaml" \
    --additional_st_model_file "/ocean/projects/cis210027p/byan/espnet-md/egs2/iwslt22_dialect/st1/exp/st_hop112_md_ag4_raw_bpe_tc1000_sp/valid.acc.ave_10best.pth" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"  "$@"
