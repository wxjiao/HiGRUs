""" Build dicts and turn sentences into indices
	Worddict is built and shared between Emotion dataset and SNLI dataset,
	but emotion labels and NLI labels are separated.
"""

import time
import re
import json
from tqdm import tqdm
import unicodedata
import argparse
from io import open
import Const
from Utils import saveToPickle, loadFrPickle,timeSince


class Dictionary:
	def __init__(self, name):
		self.name = name
		self.pre_word2count = {}
		self.rare = []
		self.word2count = {}
		self.word2index = {}
		self.index2word = {}
		self.n_words = 0
		self.max_length = 0
		self.max_dialog = 0


	def delRare(self, min_count, padunk=True):

		# collect rare words
		for w,c in self.pre_word2count.items():
			if c < min_count:
				self.rare.append(w)

		# add pad and unk
		if padunk:
			self.word2index[Const.PAD_WORD] = Const.PAD
			self.index2word[Const.PAD] = Const.PAD_WORD
			self.word2count[Const.PAD_WORD] = 1
			self.word2index[Const.UNK_WORD] = Const.UNK
			self.index2word[Const.UNK] = Const.UNK_WORD
			self.word2count[Const.UNK_WORD] = 1
			self.n_words += 2

		# index words
		for w,c in self.pre_word2count.items():
			if w not in self.rare:
				self.word2count[w] = c
				self.word2index[w] = self.n_words
				self.index2word[self.n_words] = w
				self.n_words += 1

	def addSentence(self, sentence):
		sentsplit = sentence.split(' ')
		if len(sentsplit) > self.max_length:
			self.max_length = len(sentsplit)
		for word in sentsplit:
			self.addWord(word)

	def addWord(self, word):
		if word not in self.pre_word2count:
			self.pre_word2count[word] = 1
		else:
			self.pre_word2count[word] += 1


# Preprocess of words
def unicodeToAscii(str):
	return ''.join(
		c for c in unicodedata.normalize('NFD', str)
		if unicodedata.category(c) != 'Mn'
	)

def normalizeString(str):
	str = unicodeToAscii(str.lower().strip())
	str = re.sub(r"([!?])", r" \1", str)
	str = re.sub(r"[^a-zA-Z!?]+", r" ", str)
	return str


def readUtterance(filename):
	with open(filename, encoding='utf-8') as data_file:
		data = json.loads(data_file.read())

	diadata = [[normalizeString(utter['utterance']) for utter in dialog] for dialog in data]

	emodata = [[utter['emotion'] for utter in dialog] for dialog in data]
	return diadata, emodata


def buildEmodict(dirt, phaselist, diadict, emodict):
	""" build dicts for words and emotions """
	print("Building dicts for emotion dataset...")

	max_dialog = 0
	for phase in phaselist:
		filename = dirt + phase + '.json'
		diadata, emodata  = readUtterance(filename)
		for dia, emo in zip(diadata, emodata):
			if len(dia) > max_dialog: max_dialog = len(dia)
			for d, e in zip(dia, emo):
				diadict.addSentence(d)
				emodict.addSentence(e)

	diadict.max_dialog = max_dialog

	return diadict, emodict


def indexEmo(dirt, phase, diadict, emodict, max_seq_len=60):

	filename = dirt + phase + '.json'

	diadata, emodata = readUtterance(filename)
	print('Processing file {}, length {}...'.format(filename, len(diadata)))
	diaidxs = []
	emoidxs = []
	for dia, emo in zip(diadata, emodata):
		dia_idxs = []
		emo_idxs = []
		for d, e in zip(dia, emo):
			#d_idxs = [diadict.word2index[w] if w not in diadict.rare else Const.UNK for w in d.split(' ')]
			d_idxs = [diadict.word2index[w] if w in diadict.word2index else Const.UNK for w in d.split(' ')]  # MELD and EmoryNLP not used for building vocab
			e_idxs = [emodict.word2index[e]]
			if len(d_idxs) > max_seq_len:
				dia_idxs.append(d_idxs[:max_seq_len])
			else:
				dia_idxs.append(d_idxs + [Const.PAD] * (max_seq_len - len(d_idxs)))
			emo_idxs.append(e_idxs)
		diaidxs.append(dia_idxs)
		emoidxs.append(emo_idxs)

	diafield = dict()
	diafield['feat'] = diaidxs
	diafield['label'] = emoidxs

	return diafield


def proc_emoset(dirt, phaselist, emoset, min_count, max_seq_len):
	""" Build data from emotion sets """

	diadict = Dictionary('dialogue')
	emodict = Dictionary('emotion')
	diadict, emodict = buildEmodict(dirt=dirt, phaselist=phaselist, diadict=diadict, emodict=emodict)
	diadict.delRare(min_count=min_count, padunk=True)
	emodict.delRare(min_count=0, padunk=False)
	saveToPickle(emoset + '_vocab.pt', diadict)
	print('Dialogue vocabulary (min_count={}): majority words {} rare words {}\n'.format(
		min_count, diadict.n_words, len(diadict.rare)))
	saveToPickle(emoset + '_emodict.pt', emodict)
	print('Emotions:\n {}\n {}\n'.format(emodict.word2index, emodict.word2count))

	# add the emodict for training set
	tr_diadict = Dictionary('dialogue_tr')
	tr_emodict = Dictionary('emotion_tr')
	tr_diadict, tr_emodict = buildEmodict(dirt=dirt, phaselist=['train'], diadict=tr_diadict, emodict=tr_emodict)
	tr_diadict.delRare(min_count=min_count, padunk=True)
	tr_emodict.delRare(min_count=0, padunk=False)
	saveToPickle(emoset + '_tr_emodict.pt', tr_emodict)
	print('Training set emotions:\n {}\n {}\n'.format(tr_emodict.word2index, tr_emodict.word2count))

	# index and put into fields
	Emofield = dict()
	for phase in phaselist:
		diafield = indexEmo(dirt=dirt, phase=phase, diadict=diadict, emodict=emodict, max_seq_len=max_seq_len)
		Emofield[phase] = diafield

	emo_path = emoset + '_data.pt'
	saveToPickle(emo_path, Emofield)
	print('Data written into {}!!\n'.format(emo_path))

	return 1


def main():
	''' Main function '''
	parser = argparse.ArgumentParser()
	parser.add_argument('-emoset', type=str)
	parser.add_argument('-min_count', type=int, default = 0)
	parser.add_argument('-max_seq_len', type=int, default=80)

	opt = parser.parse_args()

	print(opt, '\n')

	phaselist = ['train', 'dev', 'test']

	dirt = 'Data/' + opt.emoset + '/' + opt.emoset + '_'
	proc_emoset(dirt=dirt, phaselist=phaselist, emoset=opt.emoset, min_count=opt.min_count, max_seq_len=opt.max_seq_len)


if __name__ == '__main__':

	main()
