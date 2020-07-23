#!/bin/bash

train_model(){
    src=$1
    tgt=$2
    workloc=$3 
    modelname=$4

    fseq=$workloc/processed_data
    modeldir=$workloc/models
    mkdir -p $fseq $modeldir

    if [[ $modelname == *"ptr"* ]]; then
        useptr='--use-ptrnet'   
        fp16='' ## TODO:loss is not convergent when using fp16 for pointer network. 
    else
        useptr=''
        fp16='--fp16'
    fi
    
    if [[ $src == *"zh"* ]] || [[ $tgt == *"zh"* ]]; then
        MaxUpdates=60000    
    else
        MaxUpdates=100000
    fi

    echo "train ${modelname} NMT on $src-to-$tgt ..."   
    python train.py $fseq \
        -a transformer_wmt_en_de --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000  --seed 1 $fp16 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --max-update $MaxUpdates --exp-name "${modelname}-${src}2${tgt}" \
        --warmup-updates 16000 --warmup-init-lr '1e-07' --keep-last-epochs 80 \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --update-freq 8 --beam 2 \
        --tensorboard-logdir $modeldir/tensorboard --share-all-embeddings --consnmt $useptr \
        2>&1 | tee $modeldir/log.txt
}

get_test_BLEU(){
    src=$1
    tgt=$2
    workloc=$3 
    modelname=$4
    testclean=$5

    fseq=$workloc/processed_data
    modeldir=$workloc/models
    resdir=$workloc/result && mkdir -p $resdir 
    raw_reference=$workloc/raw/test.$tgt

    if [[ $modelname == *"ptr"* ]]; then
        useptr='--use-ptrnet'    
    else
        useptr=''
    fi

    if [[ $testclean == "1" ]]; then
        echo "test on clean dataset..."
        python generate.py $fseq -s $src -t $tgt \
            --path $modeldir/checkpoint_best_bleu.pt \
            --batch-size 20 --remove-bpe --sacrebleu \
            --decoding-path $resdir --quiet --testclean --consnmt $useptr \
            --model-overrides "{'beam':10}"
    else 
        echo "test on cons dataset..."
        python generate.py $fseq -s $src -t $tgt \
            --path $modeldir/checkpoint_best_bleu.pt \
            --batch-size 20 --remove-bpe --sacrebleu \
            --decoding-path $resdir --quiet --consnmt $useptr \
            --model-overrides "{'beam':10}"
    fi 

    detok=$workloc/detokenize.perl
    perl $detok -l $tgt < $resdir/decoding.txt > $resdir/decoding.detok
    perl $detok -l $tgt < $raw_reference > $resdir/target.detok
    cat $resdir/decoding.detok | sacrebleu $resdir/target.detok

    if [[ $testclean == "0" ]]; then
        python scripts/cal_CSR.py --src $resdir/source.txt --tgt $raw_reference \
            --hyp $resdir/decoding.txt 
    fi

}


set -e 
cd ..
export CUDA_VISIBLE_DEVICES=0
export datadir=/path/to/your/work/dir

##The processed data locates in $datadir/processed_data 
##The raw format of test reference locates in $datadir/raw/test.$tgt
##The moses detokenize scripts detokenize.perl locates in $datadir 
##you can download detokenize.perl from <https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl>


src=en
tgt=de

ModelType=leca_ptrnet  ## choice=['leca','leca_ptrnet']

echo "start to train ${ModelType} NMT model ..."
train_model $src $tgt $datadir $ModelType

echo "(1) Test on constraint-free test set"
testclean=1
get_test_BLEU $src $tgt $datadir $ModelType $testclean

echo "(2) Test on target constraints test set"
testclean=0
get_test_BLEU $src $tgt $datadir $ModelType $testclean 
