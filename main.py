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

def main():
	parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
	parser.add_argument('--dataset', type=str, default="MUTAG",
						help='name of dataset (default: MUTAG)')
	parser.add_argument('--device', type=int, default=0,
						help='which gpu to use if any (default: 0)')
	parser.add_argument('--seed', type=int, default=1,
						help='random seed (default: 42)')
	parser.add_argument('--batch_size', type=int, default=32,
						help='input batch size for training (default: 32)')
	parser.add_argument('--nhid', type=int, default=64,
						help='number of hidden feature_map dim (default: 64)')
	parser.add_argument('--nlayers', type=int, default=2,
						help='gnn layer numbers (default: 2)')
	parser.add_argument('--gat_heads', type=int, default=4,
						help='gat heads num (default: 4)')
	parser.add_argument('--epochs', type=int, default=300,
						help='number of epochs to train (default: 300)')
	parser.add_argument('--lr', type=float, default=0.001,
						help='learning rate (default: 0.001)')
	parser.add_argument('--dropout', type=float, default=0.5,
						help='dropout rate (default: 0.5)')
	parser.add_argument('--with_bn', type=bool, default=True,
						help='if with bn (default: True)')
	parser.add_argument('--with_bias', type=bool, default=True,
						help='if with bias (default: True)')
	parser.add_argument('--weight_decay', type=float, default=5e-5,
						help='weight decay of optimizer (default: 5e-5)')
	parser.add_argument('--scheduler_patience', type=int, default=50,
						help='scheduler patience (default: 50)')
	parser.add_argument('--scheduler_factor', type=float, default=0.1,
						help='scheduler factor (default: 0.1)')
	parser.add_argument('--alpha', type=float, default=1.0,
						help='the weight of distill loss(default: 0.1)')
	parser.add_argument('--tau', type=float, default=0.1,
						help='linear attention temprature(default: 0.1)')
	parser.add_argument('--early_stop', type=int, default=50,
						help='early stoping epoches (default:50)')
	parser.add_argument('--train_mode', type = str, default = "T",
										help='train mode T,S')
	parser.add_argument('--checkpoints_path', type = str, default = "checkpoints",
										help='teacher model save file path')
	parser.add_argument('--result_path', type = str, default = "results",
										help='three type models results save path')
	parser.add_argument('--backbone', type = str, default = "GCN",
										help='backbone models: GAT, GCN, GIN, SAGE')
	parser.add_argument('--runs', type = int, default = 1, help='ten-fold cross validation')
	args = parser.parse_args()

	#set up seeds and gpu device
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)    
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	checkpoints_path = f'{args.checkpoints_path}/{args.backbone}'
	result_path = f'{args.result_path}/{args.backbone}'

	if not os.path.exists(checkpoints_path):
		os.makedirs(checkpoints_path, exist_ok=True)
	if not os.path.exists(result_path):
		os.makedirs(result_path)

	path = osp.join(osp.dirname(osp.realpath(__file__)), './data', args.dataset)
	dataset = TUDataset(path, name=args.dataset, use_node_attr=True, use_edge_attr=True).shuffle()

	# pdb.set_trace()
	
	##graph features process
	x_flag = True
	if dataset[0].x == None:   ## no node features: node degree as node feature, degree as node feature
		x_flag = False
		graphs = []
		tagset = set([])
		num_features = 1
		for graph in dataset:
			x1 = list(torch.bincount(graph.edge_index[0]).numpy())
			tagset = tagset.union(set(x1))
		tagset = list(tagset)
		tag2index = {tagset[i]: i for i in range(len(tagset))}
		for graph in dataset:
			x1 = torch.bincount(graph.edge_index[0])
			x1 = (x1-torch.min(x1))/(torch.max(x1)-torch.min(x1)+0.00000001)
			graph.x = x1.view(-1, 1)
			graphs.append(graph)
		n = (len(dataset) + 9) // 10
		input_dim = num_features
		num_classes = dataset.num_classes
		del(dataset)
	else:
		n = (len(dataset) + 9) // 10
		input_dim = dataset.num_features
		num_classes = dataset.num_classes

	if args.train_mode == 'S' or args.train_mode == 'O':
		model = STnet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers, 
						gat_heads=args.gat_heads, dropout=args.dropout, with_bn=args.with_bn, with_bias=args.with_bias).to(device)

	else:
		model = Tenet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers, 
						gat_heads=args.gat_heads, dropout=args.dropout, tau=args.tau, with_bn=args.with_bn, with_bias=args.with_bias).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
															patience=args.scheduler_patience,
															factor=args.scheduler_factor,
															verbose=True)

	nll_loss = torch.nn.NLLLoss()
	mse_loss = torch.nn.MSELoss()
	test_acc_all = []
	auc_all, f1_all = [], []

	if args.train_mode == 'S':
		num_k = args.runs
	else:
		num_k = 1

	for idd in range(num_k):
		print("========================="+str(idd+1)+" on runs "+str(num_k)+"=========================")
		if x_flag:
			dataset = dataset.shuffle()
			test_dataset = dataset[:n]
			val_dataset = dataset[n:2 * n]
			train_dataset = dataset[2 * n:]
			test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
			val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
			train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
		else:
			random.shuffle(graphs)
			test_dataset = graphs[:n]
			val_dataset = graphs[n:2 * n]
			train_dataset = graphs[2 * n:]
			test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
			val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
			train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

		best_val_loss = float('inf')
		best_test_acc = 0.0
		wait = None
		best_test_acc, best_auc, best_f1 = 0.0, 0.0, 0.0

		# model training
		for epoch in range(args.epochs):
			# Training the model
			s_time = time.time()
			train_loss = 0.
			train_corrects = 0
			model.train()

			if args.train_mode == 'S':
				# load teacher model
				teacher_model = torch.load(f'{checkpoints_path}/{args.dataset}_teacher.pth').to(device)
				teacher_model.eval()

			for i, data in enumerate(train_loader):
				s = time.time()
				data = data.to(device)
				optimizer.zero_grad()

				inds = torch.tensor([data.ptr[i+1]-data.ptr[i] for i in range(data.y.shape[0])]).to(device)
				labs = torch.repeat_interleave(data.y, inds)

				if args.train_mode == 'S':
					# pdb.set_trace()
					out, st_map = model(data.x, data.edge_index, data.batch)         ##student model
					_, te_map = teacher_model(data.x, labs, data.edge_index, data.batch)
					loss_distill = mse_loss(te_map, st_map)
					loss_classification = nll_loss(out, data.y.view(-1))
					loss = loss_classification + args.alpha * loss_distill
				else:
					out, _ = model(data.x, labs, data.edge_index, data.batch)      ##teacher model input
					loss_classification = nll_loss(out, data.y.view(-1))
					loss = loss_classification

				loss.backward()
				train_loss += loss.item()
				train_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
				optimizer.step()

			train_loss /= len(train_loader)
			train_acc = train_corrects / len(train_dataset)
			scheduler.step(train_loss)

			# Validation
			val_loss = 0.
			val_corrects = 0
			model.eval()
			with torch.no_grad():
				for i, data in enumerate(val_loader):
					data = data.to(device)

					inds = torch.tensor([data.ptr[i+1]-data.ptr[i] for i in range(data.y.shape[0])]).to(device) 
					labs = torch.repeat_interleave(data.y, inds)

					if args.train_mode == 'S':
						out, st_map = model(data.x, data.edge_index, data.batch)         ##student model
						_, te_map = teacher_model(data.x, labs, data.edge_index, data.batch)
						loss_distill = mse_loss(te_map, st_map)
						loss_classification = nll_loss(out, data.y.view(-1))
						loss = loss_classification + args.alpha * loss_distill
					else:
						out, _ = model(data.x, labs, data.edge_index, data.batch)      ##teacher model input
						loss = nll_loss(out, data.y.view(-1))

					val_loss += loss.item()
					val_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

			val_loss /= len(val_loader)
			val_acc = val_corrects / len(val_dataset)

			# Test
			test_loss = 0.
			test_corrects = 0
			model.eval()
			y_preds = []
			y_tures = []
			with torch.no_grad():
				for i, data in enumerate(test_loader):
					data = data.to(device)
					inds = torch.tensor([data.ptr[i+1]-data.ptr[i] for i in range(data.y.shape[0])]).to(device) 
					labs = torch.repeat_interleave(data.y, inds) 

					if args.train_mode == 'S':
						out, _ = model(data.x, data.edge_index, data.batch)         ##student model
						loss = nll_loss(out, data.y.view(-1))
						
					else:
						out, _ = model(data.x, labs, data.edge_index, data.batch)      ##teacher model input
						loss = nll_loss(out, data.y.view(-1))

					y_preds.append(out.cpu().numpy())
					y_tures.append(data.y.view(-1).cpu().numpy())

					test_loss += loss.item()
					test_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

			test_loss /= len(test_loader)
			test_acc = test_corrects / len(test_dataset)
			auc, f1 = evaluate_func(y_preds, y_tures, args.dataset)

			log = '[*] Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}, Val Loss: {:.3f}, ' \
				  'Val Acc: {:.2f}, Test Loss: {:.3f}, Test Acc: {:.2f}, AUC:{:.3f}, F1:{:.3f}'\
				.format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, auc, f1)
			print(log)

			if best_test_acc < test_acc:
				best_test_acc = test_acc

			if best_auc < auc:
				best_auc = auc

			if best_f1 < f1:
				best_f1 = f1

			# Early-Stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				if args.train_mode == 'T':
					torch.save(model, f'{checkpoints_path}/{args.dataset}_teacher.pth')  #teacher model save
				else:
					torch.save(model, f'{checkpoints_path}/{args.dataset}_student.pth')  #student model save
				wait = 0
			else:
				wait += 1
			# early stopping
			if wait == args.early_stop:
				print('======== Early stopping! ========')
				# saving the model with best validation loss
				break

		test_acc_all.append(best_test_acc)
		auc_all.append(best_auc)
		f1_all.append(best_f1)

	pdb.set_trace()
	top_acc = np.asarray(test_acc_all)
	test_avg = np.mean(top_acc)
	test_std = np.std(top_acc)


	print("test_avg_acc: {:.5f}, test_std_acc: {:.5f}, AUC: {:.5f}, f1: {:.5f}".format(test_avg, test_std, max(auc_all), max(f1_all)))
	# pdb.set_trace()
	with open(f'{result_path}/{args.dataset}_ACC_result.txt', 'a+') as f:
		f.write(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + f" Train Mode: {args.train_mode} test_avg_acc: {test_avg:.4f}, test_std_acc: {test_std:.4f}, AUC: {max(auc_all):.4f}, F1: {max(f1_all):.4f}")
		f.write("\n")

if __name__ == '__main__':
	main()