#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

# Parameters:
SEED=$1
seed_mod="global/seed=${SEED}"

#-----------------------------------------Baseline features-----------------------------------------
#OS-experiment
#paiprun configs/main/os-baseline.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/os-baseline/${SEED}" --mods $seed_mod
#Spectrogram-experiment
#paiprun configs/main/spectrogram-baseline.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/spectrogram-baseline/${SEED}" --mods $seed_mod

#-----------------------------------------Wav2Vec2-PT-----------------------------------------------
#Local Encoder
#paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/w2v2PT-local/${SEED}" --mods $seeds_mod"&global/wav2vec2_embedding_layer=local_encoder&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml"
#Contextual Encoder
#paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/w2v2PT-contextual/${SEED}" --mods $seeds_mod"&global/wav2vec2_embedding_layer=output&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml"
#All Layers
#paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/w2v2PT-alllayers/${SEED}" --mods $seeds_mod"&global/wav2vec2_embedding_layer=enc_and_transformer"

#-----------------------------------------Wav2Vec2-LS960---------------------------------------------
ls960_mod="&global/wav2vec2_model_path=~/Models/wav2vec2/wav2vec_small_960h.pt&global/wav2vec2_dict_path=~/Models/wav2vec2"
#Local Encoder
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/w2v2LS960-local/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=local_encoder&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml$ls960_mod"
errors=$?; done
#Contextual Encoder
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/w2v2LS960-contextual/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=output&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml$ls960_mod"
errors=$?; done
#All Layers
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/w2v2LS960-alllayers/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer$ls960_mod"
errors=$?; done

#------------------------------------------Ablation--------------------------------------------------
#LSTM
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/w2v2PT-alllayers-lstm/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/dienen_config=!yaml configs/dienen/feature_learnable_combination_mean_bilstm.yaml&global/batch_size=16"
errors=$?; done
#-SpeakerNorm
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/w2v2PT-alllayers-globalnorm/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global"
errors=$?; done
#+eGeMAPS
#paiprun experiments/configs/main/w2v2-os-exps.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/w2v2PT-alllayers-egemaps/${SEED}" --mods $seeds_mod"global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global"

#---------------------------------------SOA Paper Setting---------------------------------------------
#Ravdess
#errors=1
#while (($errors!=0)); do
#paiprun configs/main/w2v2-soapaper-config.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/soapaper-ravdess/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global&global/dataset=ravdess"
#errors=$?; done
#IEMOCAP
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-soapaper-config.yaml --output_path "s3://lpepino-datasets2/is2021_experiments/using_dropout/soapaper-iemocap/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global&global/dataset=iemocap_impro"
errors=$?; done
