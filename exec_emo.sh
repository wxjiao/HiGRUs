#!bin/bash
# Var assignment
LR=2.5e-4
GPU=3
du=300
dc=300
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5
do
echo --- $Enc - $Dec $iter ---
python EmoMain.py \
-lr $LR \
-gpu $GPU \
-type higru-sf \
-d_h1 $du \
-d_h2 $dc \
-report_loss 720 \
-data_path Friends_data.pt \
-vocab_path Friends_vocab.pt \
-emodict_path Friends_emodict.pt \
-tr_emodict_path Friends_tr_emodict.pt \
-dataset Friends \
-embedding Friends_embedding.pt
done
