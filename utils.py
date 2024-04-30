import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_func(y_pred, y_ture, dataset):
	y_pred = np.vstack(y_pred)
	y_ture = np.hstack(y_ture)
	if dataset in ['MUTAG', 'PTC_FM', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
		auc = roc_auc_score(y_ture, y_pred[:,1])
	else:
		auc = 0.0
	y_pred = np.argmax(y_pred, axis=1)
	# f1_micro = f1_score(y_ture, y_pred, average='micro')
	# f1_macro = f1_score(y_ture, y_pred, average='macro')
	f1 = f1_score(y_ture, y_pred, average='weighted')

	return auc, f1