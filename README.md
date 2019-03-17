# HiGRU: Hierarchical Gated Recurrent Units for Utterance-level Emotion Recognition 

This is the Pytorch implementation of [HiGRU: Hierarchical Gated Recurrent Units for Utterance-level Emotion Recognition ](https://naacl2019.org/program/accepted/) in NAACL-2019.

## Dataset
Please find the datasets via the following links:
  1. [Friends](http://doraemon.iis.sinica.edu.tw/emotionlines)
  2. [EmotionPush](http://doraemon.iis.sinica.edu.tw/emotionlines)
  3. [IEMOCAP](https://sail.usc.edu/iemocap/)


## Run
You can run the 'exec_emo.sh' file in you **Bash** as:
'bash exec_emo.sh'

Or you can set up the model parameters yourself:
'
python EmoMain.py \
-lr 2e-4 \
-gpu 0 \
-type higru-sf \
-d_h1 300 \
-d_h2 300 \
-report_loss 720 \
-data_path Friends_data.pt \
-vocab_path Friends_vocab.pt \
-emodict_path Friends_emodict.pt \
-tr_emodict_path Friends_tr_emodict.pt \
-dataset Friends \
-embedding Friends_embedding.pt
'

