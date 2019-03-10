"""
Train on Emotion dataset
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Utils
import math


def emotrain(model, data_loader, tr_emodict, emodict, args, focus_emo):
	"""
	:data_loader input the whole field
	"""
	# start time
	time_st = time.time()
	decay_rate = args.decay

	# dataloaders
	train_loader = data_loader['train']
	dev_loader = data_loader['dev']
	feats, labels = train_loader['feat'], train_loader['label']

	lr = args.lr
	model_opt = optim.Adam(model.parameters(), lr=lr)

	# weight for loss
	weight_rate = 0.75
	if args.dataset in ['MOSI', 'IEMOCAP4v2']:
		weight_rate = 0
	weights = torch.from_numpy(loss_weight(tr_emodict, emodict, focus_emo, rate=weight_rate)).float()
	print("Dataset {} Weight rate {} \nEmotion rates {} \nLoss weights {}\n".format(
		args.dataset, weight_rate, emodict.word2count, weights))

	model.train()

	over_fitting = 0
	cur_best = -1e10
	glob_steps = 0
	report_loss = 0
	for epoch in range(1, args.epochs + 1):
		model_opt.param_groups[0]['lr'] *= decay_rate
		feats, labels = Utils.shuffle_lists(feats, labels)
		print("===========Epoch==============")
		print("-{}-{}".format(epoch, Utils.timeSince(time_st)))
		for bz in range(len(labels)):
			# tensorize a dialog list
			feat, lens = Utils.ToTensor(feats[bz], is_len=True)
			label = Utils.ToTensor(labels[bz])
			feat = Variable(feat)
			label = Variable(label)

			if args.gpu != None:
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				model.cuda(device)
				feat = feat.cuda(device)
				label = label.cuda(device)
				weights = weights.cuda(device)

			log_prob = model(feat, lens)
			#print(log_prob, label)
			loss = comput_class_loss(log_prob, label, weights)
			loss.backward()
			report_loss += loss.item()
			glob_steps += 1

			# gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

			model_opt.step()
			model_opt.zero_grad()

			if glob_steps % args.report_loss == 0:
				print("Steps: {} Loss: {} LR: {}".format(glob_steps, report_loss/args.report_loss, model_opt.param_groups[0]['lr']))
				report_loss = 0

		# validate
		pAccs = emoeval(model=model, data_loader=dev_loader, tr_emodict=tr_emodict, emodict=emodict, args=args, focus_emo=focus_emo)
		print("Validate: ACCs-F1s-WA-UWA-F1-val {}".format(pAccs))

		last_best = pAccs[-3]  # UWA
		if args.dataset in ['MOSI', 'IEMOCAP4v2']:
			last_best = pAccs[-4] # WA
		if last_best > cur_best:
			Utils.model_saver(model, args.save_dir, args.type, args.dataset)
			cur_best = last_best
			over_fitting = 0
		else:
			over_fitting += 1

		if over_fitting >= args.patience:
			break


def comput_class_loss(log_prob, target, weights):
	""" classification loss """
	loss = F.nll_loss(log_prob, target.view(target.size(0)), weight=weights, reduction='sum')
	loss /= target.size(0)

	return loss


def loss_weight(tr_ladict, ladict, focus_dict, rate=1.0):
	""" loss weights """
	min_emo = float(min([tr_ladict.word2count[w] for w in focus_dict]))
	weight = [math.pow(min_emo / tr_ladict.word2count[k], rate) if k in focus_dict
	          else 0 for k,v in ladict.word2count.items()]
	weight = np.array(weight)
	weight /= np.sum(weight)

	return weight


def emoeval(model, data_loader, tr_emodict, emodict, args, focus_emo):
	""" data_loader only input 'dev' """
	model.eval()

	# weight for loss
	weight_rate = 0.75 # eval state without weights
	if args.dataset in ['MOSI', 'IEMOCAP4v2']:
		weight_rate = 0
	weights = torch.from_numpy(loss_weight(tr_emodict, emodict, focus_emo, rate=weight_rate)).float()

	acc = np.zeros([emodict.n_words], dtype=np.long) # recall
	num = np.zeros([emodict.n_words], dtype=np.long) # gold
	preds = np.zeros([emodict.n_words], dtype=np.long) # precision, only count those in focus_emo
	focus_idx = [emodict.word2index[emo] for emo in focus_emo]

	feats, labels = data_loader['feat'], data_loader['label']
	val_loss = 0
	for bz in range(len(labels)):
		feat, lens = Utils.ToTensor(feats[bz], is_len=True)
		label = Utils.ToTensor(labels[bz])
		feat = Variable(feat)
		label = Variable(label)

		if args.gpu != None:
			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
			device = torch.device("cuda: 0")
			model.cuda(device)
			feat = feat.cuda(device)
			label = label.cuda(device)
			weights = weights.cuda(device)

		log_prob = model(feat, lens)
		# print(log_prob, label)
		# val loss
		loss = comput_class_loss(log_prob, label, weights)
		val_loss += loss.item()

		# accuracy
		emo_predidx = torch.argmax(log_prob, dim=1)
		emo_true = label.view(label.size(0))

		for lb in range(emo_true.size(0)):
			idx = emo_true[lb].item()
			num[idx] += 1
			if emo_true[lb] == emo_predidx[lb]:
				acc[idx] += 1
			# count for precision
			if emo_true[lb] in focus_idx and emo_predidx[lb] in focus_idx:
				preds[emo_predidx[lb]] += 1

	pacc = [acc[i] for i in range(emodict.n_words) if emodict.index2word[i] in focus_emo]
	pnum = [num[i] for i in range(emodict.n_words) if emodict.index2word[i] in focus_emo]
	pwACC = sum(pacc) / sum(pnum) * 100
	ACCs = [np.round(acc[i] / num[i] * 100, 2) if num[i] != 0 else 0 for i in range(emodict.n_words)]
	pACCs = [ACCs[i] for i in range(emodict.n_words) if emodict.index2word[i] in focus_emo]
	paACC = sum(pACCs) / len(pACCs)
	pACCs = [ACCs[emodict.word2index[w]] for w in focus_emo] # recall

	TP = [acc[emodict.word2index[w]] for w in focus_emo]
	pPREDs = [preds[emodict.word2index[w]] for w in focus_emo]
	pPRECs = [np.round(tp/p*100,2) if p>0 else 0 for tp,p in zip(TP,pPREDs)] # precision
	pF1s = [np.round(2*r*p/(r+p),2) if r+p>0 else 0 for r,p in zip(pACCs,pPRECs)]
	F1 = sum(pF1s)/len(pF1s)

	Total = [pACCs] + [pF1s] + [np.round(pwACC,2), np.round(paACC,2), np.round(F1,2), -val_loss]

	model.train()

	return Total
