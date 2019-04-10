# HiGRU: Hierarchical Gated Recurrent Units for Utterance-level Emotion Recognition 

This is the Pytorch implementation of [HiGRU: Hierarchical Gated Recurrent Units for Utterance-level Emotion Recognition ](https://arxiv.org/abs/1904.04446) in NAACL-2019.

## Dataset
Please find the datasets via the following links:
  1. [Friends](http://doraemon.iis.sinica.edu.tw/emotionlines)
  2. [EmotionPush](http://doraemon.iis.sinica.edu.tw/emotionlines)
  3. [IEMOCAP](https://sail.usc.edu/iemocap/)


## Run
#### Data Preprocessing
For each dataset, we need to preprocess it using the `Preprocess.py` file as:
```
python Preprocess.py -emoset Friends -min_count 2 -max_length 60
```
The arguments `-emoset`, `-min_count`, and `-max_length` represent the dataset name, the minimum frequency of words when building
the vocabulary, and the max_length for padding or truncating sentences.

#### Train
You can run the `exec_emo.sh` file in you **Bash** as:
```
bash exec_emo.sh
```

Or you can set up the model parameters yourself:
```
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
```

The implementation supports both CPU and GPU (but only one GPU), you need to specify the device number of GPU in your arguments otherwise the model will be trained in CPU. There are **three** modes in this implementation, i.e., `higru`, `higru-f`, and `higru-sf`, as described in the paper. You can select one of them by the argument `-type`. The default sizes of the hidden states in the GRUs are 300, but smaller values also work well (larger ones may result in over-fitting). You have to load in the data produced by the `Preprocess.py` file, e.g., including `Friends_data.pt`, `Friends_vocab.pt`, `Friends_emodict.pt`, and `Friends_tr_emodict.pt`, as well as the dataset name `Friends`. The argument `-embedding` is optional that you can load in the embeddings saved by the first run or the implementation will initialize it every time (which is time-consuming).
There are some other arguments in the `EmoMain.py` file, e.g., the decay rate for learning rate, the patience for early stopping. You can find out and change them if necessary.


