""" functions, layers, and architecture of HiGRU """
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import Const


# Dot-product attention
def get_attention(q, k, v, attn_mask=None):
	"""
	:param : (batch, seq_len, seq_len)
	:return: (batch, seq_len, seq_len)
	"""
	attn = torch.matmul(q, k.transpose(1, 2))
	if attn_mask is not None:
		attn.data.masked_fill_(attn_mask, -1e10)

	attn = F.softmax(attn, dim=-1)
	output = torch.matmul(attn, v)
	return output, attn


# Get mask for attention
def get_attn_pad_mask(seq_q, seq_k):
	assert seq_q.dim() == 2 and seq_k.dim() == 2

	pad_attn_mask = torch.matmul(seq_q.unsqueeze(2).float(), seq_k.unsqueeze(1).float())
	pad_attn_mask = pad_attn_mask.eq(Const.PAD)  # b_size x 1 x len_k
	#print(pad_attn_mask)

	return pad_attn_mask.cuda(seq_k.device)


# Pad for utterances with variable lengths and maintain the order of them after GRU
class GRUencoder(nn.Module):
	def __init__(self, d_emb, d_out, num_layers):
		super(GRUencoder, self).__init__()
		# default encoder 2 layers
		self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
		                  bidirectional=True, num_layers=num_layers, dropout=0.3)

	def forward(self, sent, sent_lens):
		"""
		:param sent: torch tensor, batch_size x seq_len x d_rnn_in
		:param sent_lens: numpy tensor, batch_size x 1
		:return:
		"""
		device = sent.device
		# seq_len x batch_size x d_rnn_in
		sent_embs = sent.transpose(0,1)

		# sort by length
		s_lens, idx_sort = np.sort(sent_lens)[::-1], np.argsort(-sent_lens)
		idx_unsort = np.argsort(idx_sort)

		idx_sort = torch.from_numpy(idx_sort).cuda(device)
		s_embs = sent_embs.index_select(1, Variable(idx_sort))

		# padding
		s_lens = s_lens.copy()
		sent_packed = pack_padded_sequence(s_embs, s_lens)
		sent_output = self.gru(sent_packed)[0]
		sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

		# unsort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))

		# batch x seq_len x 2*d_out
		output = sent_output.transpose(0,1)

		return output


# Utterance encoder with three types: higru, higru-f, and higru-sf
class UttEncoder(nn.Module):
	def __init__(self, d_word_vec, d_h1, type):
		super(UttEncoder, self).__init__()
		self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1)
		self.d_input = 2 * d_h1
		self.model = type
		if self.model == 'higru-f':
			self.d_input = 2 * d_h1 + d_word_vec
		if self.model == 'higru-sf':
			self.d_input = 4 * d_h1 + d_word_vec
		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h1),
			nn.Tanh()
		)

	def forward(self, sents, lengths, sa_mask=None):
		"""
		:param sents: batch x seq_len x 2*d_h1
		:param lengths: numpy array 1 x batch
		:return: batch x d_h1
		"""
		w_context = self.encoder(sents, lengths)
		combined = w_context

		if self.model == 'higru-f':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			combined = [w_lcont, sents, w_rcont]
			combined = torch.cat(combined, dim=-1)
		if self.model == 'higru-sf':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
			sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
			combined = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
			combined = torch.cat(combined, dim=-1)

		output1 = self.output1(combined)
		output = torch.max(output1, dim=1)[0]

		return output


# The overal HiGRU model with three types: HiGRU, HiGRU-f, HiGRU-sf
class HiGRU(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru'):
		super(HiGRU, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2

		# load word2vec
		self.embeddings = embedding

		self.uttenc = UttEncoder(d_word_vec, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=True)
		self.d_input = 2 * d_h2
		if self.model == 'higru-f':
			self.d_input = 2 * d_h2 + d_h1
		if self.model == 'higru-sf':
			self.d_input = 4 * d_h2 + d_h1

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

	def forward(self, sents, lens):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		s_embed = self.uttenc(w_embed, lens, sa_mask)
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context
		if self.model == 'higru-f':
			s_lcont, s_rcont = s_context.chunk(2,-1)
			Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
			Combined = torch.cat(Combined, dim=-1)
		if self.model == 'higru-sf':
			s_lcont, s_rcont = s_context.chunk(2, -1)
			SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
			SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
			Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
			Combined = torch.cat(Combined, dim=-1)

		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)

		output = self.classifier(output1)
		pred_scores = F.log_softmax(output, dim=1)

		return pred_scores
