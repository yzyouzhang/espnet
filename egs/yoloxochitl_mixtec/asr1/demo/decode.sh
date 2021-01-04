#  copyright 2020 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
nj=1

# dataset related
decode_config=conf/decode.yaml
lmtag=mixtec_underlying_full_reserve        # tag for managing LMs
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
dumpdir=dump   # directory to dump full features
tag=mixtec_underlying_full_reserve-conformer-specaug-sp
dict=data/lang_char/train_mixtec_underlying_full_reserve_sp_unigram150_units.txt
bpemodel=data/lang_char/train_mixtec_underlying_full_reserve_sp_unigram150
lmexpname=train_rnnlm_pytorch_mixtec_underlying_full_reserve_unigram150
lmexpdir=exp/${lmexpname}
expname=train_mixtec_underlying_full_reserve_sp_pytorch_${tag}
expdir=exp/${expname}
output_dir=elan

annotation_dir=demo
sound_dir=demo
recog_files=demo/blank_files.csv

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


recog_set="demo"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Prepare Data"
    python demo/scripts/data_prep_for_botany_mono_blank.py -a ${annotation_dir} -t data/${recog_set} -s ${sound_dir} -i ${recog_files} --lang mixtec
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${recog_set}; do
        utils/fix_data_dir.sh data/${x}
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
                                  data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/deltafalse; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${rtask}/feats.scp data/train_mixtec_underlying_full_reserve_sp/cmvn.ark exp/dump_feats/recog/${rtask} \
                ${feat_recog_dir}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dump Data for Decoding"
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/deltafalse
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
                     data/${rtask} ${dict} > ${feat_recog_dir}/data_unigram150.json
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    nj=${nj}

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/deltafalse

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_unigram150.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_unigram150.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}
            --rnnlm ${lmexpdir}/rnnlm.model.best

        decode_hyp.sh --bpe 150 --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: ELAN Importable Files"
    mkdir -p ${output_dir}
    decode_dir=decode_${recog_set}_$(basename ${decode_config%.*})_${lmtag}
    python demo/scripts/ElanImport.py data/${recog_set} ${expdir}/${decode_dir}/hyp.wrd.trn ${output_dir}
fi

echo "Finished all stages"

