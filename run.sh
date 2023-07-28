#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

stage=1
stop_stage=4

FAIRSEQ_ROOT=${PWD}/env/fairseq
export PYTHONPATH=$PWD:$FAIRSEQ_ROOT:$PYTHONPATH

export decode_cmd="run.pl"
. ./local/swbd/path.sh


set -e
set -u
set -o pipefail

exp_suffix=swbd
manifest_dir=data/swbd


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Continue pretraining wav2vec2 model"
    fairseq-hydra-train \
        dataset.train_subset=train \
        dataset.valid_subset=dev \
        task.data=${PWD}/${manifest_dir} \
        checkpoint.finetune_from_model=${PWD}/data/model/wav2vec_small.pt \
        hydra.run.dir=${PWD}/exp/pretrain/${exp_suffix}/continue_pretrain_20k \
        --config-dir ${PWD}/config/pretraining \
        --config-name wav2vec2_base_finetune_20k.yaml > logs/train_continue_pretrain_20k_${exp_suffix}.log 2>&1
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	echo "stage 2: online PL (DPL)"
        fairseq-hydra-train \
            distributed_training.distributed_port=-1 \
            task.data=${PWD}/${manifest_dir} \
            dataset.train_subset=train_semi \
            dataset.valid_subset=dev_semi \
            +criterion.wer_kenlm_model=${PWD}/${manifest_dir}/lm/lm_fglarge.bin \
            +criterion.wer_lexicon=${PWD}/${manifest_dir}/lm/char_lexicon.txt \
	        +criterion.wer_lm_weight=2 \
            +criterion.wer_word_score=1 \
            model.w2v_path=${PWD}/exp/pretrain/${exp_suffix}/continue_pretrain_20k/checkpoints/checkpoint_best.pt \
            hydra.run.dir=${PWD}/exp/finetune/${exp_suffix}/dpl \
            --config-dir ${PWD}/config/finetuning \
            --config-name dpl >logs/dpl.log 2>&1
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
	echo "stage 3: offline PL (with UCF)"

    suffix=dpl_ucf
    for iter in 1 2;do
        # define exp_name
        if [ ${iter} -eq 1 ];then
            exp_name=dpl
        else
            exp_name=train_self_training_${suffix}_iter_"$((iter-1))"
        fi
        # decode
        decode_nj=16
        ngpu=8
        checkpoint_path=${PWD}/exp/finetune/${exp_suffix}/${exp_name}/checkpoints/checkpoint_best.pt
        pseudo_label_decode_dir=${PWD}/${manifest_dir}/pseudo_label/results_${suffix}/iter_${iter}
        pseudo_label_dir=${PWD}/${manifest_dir}/pseudo_label/pseudo_label_${suffix}/iter_${iter}
        pseudo_label_init_dir=${PWD}/${manifest_dir}/pseudo_label/pseudo_label_init
        log_name=decode_pseudo_label_${exp_name}_${suffix}_${exp_suffix}

        for subset in train;do
            python3 local/split_decode_data.py --datadir ${PWD}/${manifest_dir} --subset ${subset} --num ${decode_nj}
            mkdir -p $pseudo_label_decode_dir/split${decode_nj} || true
            ${decode_cmd} JOB=1:${decode_nj} $pseudo_label_decode_dir/split${decode_nj}/log/decode.JOB.log \
            python3 ${PWD}/local/decode/infer_ucf.py ${PWD}/${manifest_dir}/split${decode_nj} --task audio_pretraining \
                --nbest 1 --path ${checkpoint_path} --gen-subset ${subset}_JOB \
                --results-path ${pseudo_label_decode_dir}/split${decode_nj} --w2l-decoder dropoutkenlm --lexicon ${PWD}/${manifest_dir}/lm/char_lexicon.txt \
                --lm-model ${PWD}/${manifest_dir}/lm/lm_fglarge.bin --lm-weight 1 --word-score 1 --sil-weight 0 \
                --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter --beam 50 --job_id JOB --gpu_num ${ngpu}  > new_logs/${log_name} 2>&1
            python3 local/combine_decode_results.py --datadir ${PWD}/${manifest_dir} --subset ${subset} --num ${decode_nj} \
                --decodedir ${pseudo_label_decode_dir} \
                --files "hypo.word-0-checkpoint_best.pt-${subset}.txt,hypo.word-1-checkpoint_best.pt-${subset}.txt,hypo.word-2-checkpoint_best.pt-${subset}.txt,hypo.word-3-checkpoint_best.pt-${subset}.txt,hypo.word.score-0-checkpoint_best.pt-${subset}.txt,hypo.word.score-1-checkpoint_best.pt-${subset}.txt,hypo.word.score-2-checkpoint_best.pt-${subset}.txt,hypo.word.score-3-checkpoint_best.pt-${subset}.txt,ref.word-0-checkpoint_best.pt-${subset}.txt,ref.word-1-checkpoint_best.pt-${subset}.txt,ref.word-2-checkpoint_best.pt-${subset}.txt,ref.word-3-checkpoint_best.pt-${subset}.txt"
            python3 local/filter_ucf.py --dir ${pseudo_label_decode_dir}
        done

        threshold=11.7
        pseudo_label_dir=${PWD}/${manifest_dir}/pseudo_label/pseudo_label_${suffix}/iter_${iter}
        pseudo_label_dir=${pseudo_label_dir}_threshold_${threshold}
        # transform decoding results to pseudo labels
        rm -r ${pseudo_label_dir} || true
        mkdir -p ${pseudo_label_dir}
        cp ${pseudo_label_init_dir}/* ${pseudo_label_dir}
        python3 local/parse_pseudo_label_filter_ucf.py \
            --threshold ${threshold} --uncertainty-weight 10 --length-bonus 0.1 \
            --trn ${pseudo_label_decode_dir}/hypo.word.score-checkpoint_best.pt-train.txt  \
            --tsv ${pseudo_label_dir}/train.tsv --wrd ${pseudo_label_dir}/train.wrd --ltr ${pseudo_label_dir}/train.ltr
        # train on pseudo labels

        fairseq-hydra-train \
            distributed_training.distributed_port=-1 \
            task.data=${pseudo_label_dir} \
            dataset.train_subset=train \
            dataset.valid_subset=dev \
            +criterion.wer_kenlm_model=${PWD}/${manifest_dir}/lm/lm_fglarge.bin \
            +criterion.wer_lexicon=${PWD}/${manifest_dir}/lm/char_lexicon.txt \
	        +criterion.wer_lm_weight=2 \
            +criterion.wer_word_score=1 \
	        checkpoint.finetune_from_model=${PWD}/exp/finetune/${exp_suffix}/${exp_name}/checkpoints/checkpoint_best.pt \
            model.w2v_path=/home/zhuhan/project/fairseq/fairseq_20210413/examples/wav2vec/data/model/wav2vec_small.pt \
            hydra.run.dir=${PWD}/exp/finetune/${exp_suffix}/train_self_training_${suffix}_iter_${iter} \
            --config-dir ${PWD}/config/finetuning \
            --config-name ctc_finetune > logs/train_self_training_${suffix}_iter_${iter}_${exp_suffix}.log 2>&1
    done

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding with kenlm decoder"
    exp_name=dpl_ucf
    checkpoint=checkpoint_best
    lm_weight=2
    word_score=1
    checkpoint_path=${PWD}/exp/finetune/${exp_suffix}/${exp_name}/checkpoints/${checkpoint}.pt
    log_name=decode_${exp_name}_${checkpoint}_lw_${lm_weight}_ws_${word_score}
    for subset in dev eval2000;do
        python3 ${FAIRSEQ_ROOT}/examples/speech_recognition/infer.py ${PWD}/${manifest_dir} --task audio_pretraining \
            --nbest 1 --path ${checkpoint_path} --gen-subset $subset \
            --results-path ${PWD}/results/${log_name} --w2l-decoder kenlm --lexicon ${PWD}/${manifest_dir}/lm/char_lexicon.txt \
            --lm-model ${PWD}/${manifest_dir}/lm/lm_fglarge.bin --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 \
            --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter --beam 50  >> ${log_name} 2>&1
	    mkdir -p ${PWD}/results/${log_name}/decode_${subset} || true
	    rm -r ${PWD}/results/${log_name}/decode_${subset}/scoring || true
	    python3 local/swbd/swbd_replace_idx.py --idx ${manifest_dir}/${subset}.idx --input-trn ${PWD}/results/${log_name}/hypo.word-${checkpoint}.pt-${subset}.txt --output-trn ${PWD}/results/${log_name}/decode_${subset}/hyp.wrd.trn
	    python3 local/swbd/swbd_replace_idx.py --idx ${manifest_dir}/${subset}.idx --input-trn ${PWD}/results/${log_name}/ref.word-${checkpoint}.pt-${subset}.txt --output-trn ${PWD}/results/${log_name}/decode_${subset}/ref.wrd.trn
        local/swbd/score_sclite.sh data/swbd/${subset} results/${log_name}/decode_${subset} >> logs/${log_name} 2>&1
    done
fi