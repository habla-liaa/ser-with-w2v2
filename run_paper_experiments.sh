#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

# Parameters:
SEED=$1
OUTPUT_PATH=$2
seed_mod="global/seed=${SEED}"

#-----------------------------------------Baseline features-----------------------------------------
#none-egemaps
errors=1
while (($errors!=0)); do
paiprun configs/main/os-baseline.yaml --output_path "${OUTPUT_PATH}/none-egemaps/${SEED}" --mods $seed_mod
errors=$?; done
#none-spectrogram
errors=1
while (($errors!=0)); do
paiprun configs/main/spectrogram-baseline.yaml --output_path "${OUTPUT_PATH}/none-spectrogram/${SEED}" --mods $seed_mod
errors=$?; done
#-----------------------------------------Wav2Vec2-PT-----------------------------------------------
#w2v2PT-local
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-local/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=local_encoder&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml"
errors=$?; done
#w2v2PT-contextual
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-contextual/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=output&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml"
errors=$?; done
#w2v2PT-alllayers
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-alllayers/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer"
errors=$?; done
#-----------------------------------------Wav2Vec2-LS960---------------------------------------------
ls960_mod="&global/wav2vec2_model_path=~/Models/wav2vec2/wav2vec_small_960h.pt&global/wav2vec2_dict_path=~/Models/wav2vec2"
#w2v2LS960-local
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2LS960-local/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=local_encoder&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml$ls960_mod"
errors=$?; done
#w2v2LS960-contextual
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2LS960-contextual/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=output&global/dienen_config=!yaml configs/dienen/mean_mlp.yaml$ls960_mod"
errors=$?; done
#w2v2LS960-alllayers
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2LS960-alllayers/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer$ls960_mod"
errors=$?; done

#------------------------------------------Ablation--------------------------------------------------
#w2v2PT-alllayers-lstm
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-alllayers-lstm/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/dienen_config=!yaml configs/dienen/feature_learnable_combination_mean_bilstm.yaml&global/batch_size=16"
errors=$?; done
#w2v2PT-alllayers-global
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-alllayers-global/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global"
errors=$?; done
#w2v2PT-fusion
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-os-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-fusion/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global"
errors=$?; done

#---------------------------------------Issa et al. Paper Setting---------------------------------------------
#Ravdess
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-soapaper-config.yaml --output_path "${OUTPUT_PATH}/issa-setup-ravdess/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global&global/dataset=ravdess"
errors=$?; done
#IEMOCAP
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-soapaper-config.yaml --output_path "${OUTPUT_PATH}/issa-setup-iemocap/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global&global/dataset=iemocap_impro"
errors=$?; done
