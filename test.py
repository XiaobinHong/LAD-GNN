import os.path as osp
import time
import torch
import os 
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Dataset

from GNN_models.base_model import STnet, Tenet
from utils import evaluate_func
from torch_geometric.utils import degree

import numpy as np
import pdb
import datetime
import random
import argparse

def Test(test_loader, device, args):
	if args.train_mode == 'S':
		model  = torch.load(f'{checkpoints_path}/{args.dataset}_student.pth').to(device)
	elif args.train_mode == 'O':
		model  = torch.load(f'{checkpoints_path}/{args.dataset}_original.pth').to(device)
	else:
		model  = torch.load(f'{checkpoints_path}/{args.dataset}_teacher.pth').to(device)

	model.eval()
	y_preds, y_true = [], []
	test_loss, test_corrects = 0., 0
	nll_loss = torch.nn.NLLLoss()
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			data = data.to(device)
			inds = torch.tensor([data.ptr[i+1]-data.ptr[i] for i in range(data.y.shape[0])]).to(device) 
			labs = torch.repeat_interleave(data.y, inds)
			if args.train_mode == 'O' or args.train_mode == 'S':
				out, _ = model(data.x, data.edge_index, data.batch)
				loss = nll_loss(out, data.y.view(-1))
			else:
				out, _ = model(data.x, labs, data.edge_index, data.batch)
				loss = nll_loss(out, data.y.view(-1))

			y_preds.append(out.cpu().numpy())
			y_true.append(out.cpu().numpy())

			test_loss += loss.item()
			test_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

	test_loss /= len(test_loader)
	test_acc = test_corrects / len(test_loader)
	auc, f1 = evaluate_func(y_preds, y_true, args.dataset)
	print(f"Dataset: {args.dataset}, Test Accuracy: {test_acc:.3f}, Test AUC: {auc:.3f}, Test F1 Score: {f1:.3f}")
	if auc == 0:
		print(f"{args.dataset} is not a binary classification dataset for AUC.")