"""
main function for wordCNN or +charCNN training
"""
import os
import argparse
import Utils
import Const
from Preprocess import Dictionary # import the object for pickle loading
from Modules import *
from EmoTrain import emotrain, emoeval
from datetime import datetime
import math
import time
#str(datetime.now())
#'2011-05-03 17:45:35.177000'


def main():
	'''Main function'''

	parser = argparse.ArgumentParser()

	# learning
	parser.add_argument('-lr', type=float, default=2.5e-4) # 2.5e-4 for Friends and EmotionPush, 1e-4 for IEMOCAP
	parser.add_argument('-decay', type=float, default=math.pow(0.5, 1/20)) # math.pow(0.5, 1/20)
	parser.add_argument('-epochs', type=int, default=200)
	parser.add_argument('-patience', type=int, default=10,
	                    help='patience for early stopping')
	parser.add_argument('-save_dir', type=str, default="snapshot",
	                    help='where to save the models')
	# data
	parser.add_argument('-dataset', type=str, default='Friends',
	                    help='dataset')
	parser.add_argument('-data_path', type=str, required = True,
	                    help='data path')
	parser.add_argument('-vocab_path', type=str, required=True,
	                    help='global vocabulary path')
	parser.add_argument('-emodict_path', type=str, required=True,
	                    help='emotion label dict path')
	parser.add_argument('-tr_emodict_path', type=str, default=None,
	                    help='training set emodict path')
	parser.add_argument('-max_seq_len', type=int, default=80, # 80 for emotion
	                    help='the sequence length')
	# model
	parser.add_argument('-type', type=str, default='higru',
	                    help='choose the low encoder')
	parser.add_argument('-d_word_vec', type=int, default=300,
	                    help='the word embeddings size')
	parser.add_argument('-d_h1', type=int, default=300,
	                    help='the hidden size of rnn1')
	parser.add_argument('-d_h2', type=int, default=300,
	                    help='the hidden size of rnn1')
	parser.add_argument('-d_fc', type=int, default=100,
	                    help='the size of fc')
	parser.add_argument('-gpu', type=str, default=None,
	                    help='gpu: default 0')
	parser.add_argument('-embedding', type=str, default=None,
	                    help='filename of embedding pickle')
	parser.add_argument('-report_loss', type=int, default=720,
	                    help='how many steps to report loss')

	args = parser.parse_args()
	print(args, '\n')

	# load vocabs
	print("Loading vocabulary...")
	worddict = Utils.loadFrPickle(args.vocab_path)
	print("Loading emotion label dict...")
	emodict = Utils.loadFrPickle(args.emodict_path)
	print("Loading review tr_emodict...")
	tr_emodict = Utils.loadFrPickle(args.tr_emodict_path)

	# load field
	print("Loading field...")
	field = Utils.loadFrPickle(args.data_path)
	test_loader = field['test']

	# word embedding
	print("Initializing word embeddings...")
	embedding = nn.Embedding(worddict.n_words, args.d_word_vec, padding_idx=Const.PAD)
	if args.d_word_vec == 300:
		if args.embedding != None and os.path.isfile(args.embedding):
			np_embedding = Utils.loadFrPickle(args.embedding)
		else:
			np_embedding = Utils.load_pretrain(args.d_word_vec, worddict, type='word2vec')
			Utils.saveToPickle(args.dataset + '_embedding.pt', np_embedding)
		embedding.weight.data.copy_(torch.from_numpy(np_embedding))
	embedding.weight.requires_grad = False

	# model type
	model = HiGRU(d_word_vec=args.d_word_vec,
	              d_h1=args.d_h1,
	              d_h2=args.d_h2,
	              d_fc=args.d_fc,
	              emodict=emodict,
	              worddict=worddict,
	              embedding=embedding,
	              type=args.type)

	# Choose focused emotions
	focus_emo = Const.four_emo
	if args.dataset == 'IEMOCAP4v2':
		focus_emo = Const.four_iem
	print("Focused emotion labels {}".format(focus_emo))

	# train, adjust lr w.r.t. hidden size
	#args.lr *= math.pow(300/args.d_hidden_low * 300/args.d_hidden_up, 0.5)
	#if args.load_model:
	#	args.lr /= 2
	emotrain(model=model,
	         data_loader=field,
	         tr_emodict=tr_emodict,
	         emodict=emodict,
	         args=args,
	         focus_emo=focus_emo)

	# test
	print("Load best models for testing!")

	model = Utils.model_loader(args.save_dir, args.type, args.dataset)
	pAccs = emoeval(model=model,
	                data_loader=test_loader,
	                tr_emodict=tr_emodict,
	                emodict=emodict,
	                args=args,
	                focus_emo=focus_emo)
	print("Test: ACCs-F1s-WA-UWA-F1-val {}".format(pAccs))

	# record the test results
	record_file = '{}/{}_{}.txt'.format(args.save_dir, args.type, args.dataset)
	if os.path.isfile(record_file):
		f_rec = open(record_file, "a")
	else:
		f_rec = open(record_file, "w")
	f_rec.write("{} - {} - {}\t:\t{}\n".format(datetime.now(), args.d_h1, args.lr, pAccs))
	f_rec.close()


if __name__ == '__main__':
	main()
