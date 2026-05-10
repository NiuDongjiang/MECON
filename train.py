from datetime import datetime
import time
import argparse
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import Moe
import Str_expert
from KG_expert import KG_expert
import custom_loss
from data_preprocessing_t_b_b import DrugDataset, DrugDataLoader
from get_args import config
import warnings
import pickle as pkl
from tqdm import tqdm
import torch.multiprocessing as mp
import os

warnings.filterwarnings('ignore', category=UserWarning)
mp.set_sharing_strategy("file_system")


######################### Parameters ######################
dataset_name = config['dataset_name']
params = config['params']
lr = params['lr']
n_epochs = 50
batch_size = params['batch_size']
weight_decay = params['weight_decay']
neg_samples = params['neg_samples']
data_size_ratio = params['data_size_ratio']
device = 'cuda:2' if torch.cuda.is_available() and params['use_cuda'] else 'cpu'
for it in [30]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hyper_num', type=int, default=4)
    parser.add_argument('--ddi_types', type=int, default=86, help='86 for drugbank, 963 for twosides')
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--bal_beta', type=float, default=1e-2)
    parser.add_argument('--moe_topk', type=int, default=2)
    parser.add_argument('--pareto_iters', type=int, default=it)#30
    args = parser.parse_args()

    print(dataset_name, params)

    n_atom_feats = 55
    rel_total = 86
    kge_dim = 128

    fold = 1
    dataset_n = 'drugbank'
    rst_file = f'model_pkl/{dataset_n}_moe_pareto/'
    rst_mol = f'model_pkl/{dataset_n}_mol/'
    rst_kg = f'model_pkl/{dataset_n}_kg/'
    save_dir = os.path.join(rst_file, f'fold{fold}')
    save_mol = os.path.join(rst_mol, f'fold{fold}')
    save_kg = os.path.join(rst_kg, f'fold{fold}')
    os.makedirs(save_dir, exist_ok=True)

    pkl_name = os.path.join(save_dir, f'train_moe_pareto_{fold}_hyper{it}.pkl')
    pkl_mol = os.path.join(save_mol, f'train_mol_{fold}.pkl')
    pkl_kg = os.path.join(save_kg, f'train_kg_{fold}.pkl')

    with open(f'dataset/trans/{dataset_n}/Drkg_TransE_Emb.pkl', 'rb') as file:
        Trans_emb = pkl.load(file)
    Trans_emb_HG = Trans_emb.to(device)
    Trans_emb_HY = Trans_emb.to(device)


    ######################### Dataset ######################
    def split_train_valid(data, fold, val_ratio=0.2):
        data = np.array(data)
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
        train_tup = data[train_index]
        val_tup = data[val_index]
        train_tup = [(tup[0], tup[1], int(tup[2])) for tup in train_tup]
        val_tup = [(tup[0], tup[1], int(tup[2])) for tup in val_tup]
        return train_tup, val_tup


    if 'drugbank' not in dataset_name:
        df_ddi_train = pd.read_csv(config[dataset_name]["trans_ddi_train"])
        df_ddi_test = pd.read_csv(config[dataset_name]["trans_ddi_test"])
        df_ddi_valid = pd.read_csv(config[dataset_name]["trans_ddi_valid"])

        train_tup = [(h, t, r) for h, t, r in
                     zip(df_ddi_train['drugbank_id_1'], df_ddi_train['drugbank_id_2'], df_ddi_train['label'])]
        val_tup = [(h, t, r) for h, t, r in
                   zip(df_ddi_valid['drugbank_id_1'], df_ddi_valid['drugbank_id_2'], df_ddi_valid['label'])]
        test_tup = [(h, t, r) for h, t, r in
                    zip(df_ddi_test['drugbank_id_1'], df_ddi_test['drugbank_id_2'], df_ddi_test['label'])]
    else:
        df_ddi_train = pd.read_csv(config[dataset_name]["trans_ddi_train"])
        df_ddi_test = pd.read_csv(config[dataset_name]["trans_ddi_test"])

        train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
        train_tup, val_tup = split_train_valid(train_tup, fold, val_ratio=0.2)
        test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

    train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
    val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
    test_data = DrugDataset(test_tup, disjoint_split=False)

    print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_data_loader = DrugDataLoader(val_data, batch_size=batch_size * 3, num_workers=2)
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3, num_workers=2)


    ######################### Pareto Solver ######################
    class ParetoMGDASolver:


        def __init__(self, max_iter=30, tol=1e-6, eps=1e-12):
            self.max_iter = max_iter
            self.tol = tol
            self.eps = eps

        def _flatten_grads(self, grads, params):
            flat = []
            for g, p in zip(grads, params):
                if g is None:
                    flat.append(torch.zeros_like(p).reshape(-1))
                else:
                    flat.append(g.reshape(-1))
            if len(flat) == 0:
                return None
            return torch.cat(flat)

        def _project_simplex(self, y):

            sorted_y, _ = torch.sort(y, descending=True)
            cssv = torch.cumsum(sorted_y, dim=0) - 1
            ind = torch.arange(1, y.numel() + 1, device=y.device, dtype=y.dtype)
            cond = sorted_y - cssv / ind > 0
            rho = ind[cond][-1]
            theta = cssv[cond][-1] / rho
            return torch.clamp(y - theta, min=0.0)

        def solve(self, losses, model):
            params = [p for p in model.parameters() if p.requires_grad]
            grad_list = []

            for loss in losses:
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    allow_unused=True
                )
                flat_grad = self._flatten_grads(grads, params)
                norm = flat_grad.norm(p=2).clamp_min(self.eps)
                grad_list.append(flat_grad / norm)

            G = torch.stack(grad_list, dim=0)          # [T, D]
            GG = torch.matmul(G, G.t())                # [T, T]
            T = GG.size(0)

            alpha = torch.full((T,), 1.0 / T, device=GG.device, dtype=GG.dtype)

            for _ in range(self.max_iter):
                grad_alpha = 2.0 * torch.mv(GG, alpha)
                new_alpha = self._project_simplex(alpha - 0.1 * grad_alpha)

                if torch.norm(new_alpha - alpha, p=2) < self.tol:
                    alpha = new_alpha
                    break
                alpha = new_alpha

            alpha = alpha / alpha.sum().clamp_min(self.eps)
            return alpha.detach()


    ######################### Forward Utilities ######################
    def _forward_single_branch(tri, model, model_mol, model_kg):
        tri = [x.to(device) if hasattr(x, "to") else x for x in tri]
        (h_data, h_data_fin, h_data_desc,
         t_data, t_data_fin, t_data_desc,
         rels, h_data_edge, t_data_edge,
         Page_SUBKG_data1, Page_SUBKG_data2,
         RW_SUBKG_data1, RW_SUBKG_data2, b_graph) = tri

        drug_index1 = h_data.drug_index
        drug_index2 = t_data.drug_index
        DDI_type_index = rels.squeeze(0)

        # Freeze both experts during Pareto gating training
        with torch.no_grad():
            score_mol, h_pool, t_pool = model_mol(
                h_data, h_data_fin, h_data_desc,
                t_data, t_data_fin, t_data_desc,
                rels, h_data_edge, t_data_edge, b_graph
            )

            score_kg, _, _, _, KG_d1_feat, KG_d2_feat = model_kg(
                drug_index1, drug_index2, DDI_type_index,
                Trans_emb_HG, Trans_emb_HY,
                Page_SUBKG_data1, Page_SUBKG_data2,
                RW_SUBKG_data1, RW_SUBKG_data2
            )

        moe_out = model(
            score_mol=score_mol,
            score_kg=score_kg.squeeze(-1),
            h_pool=h_pool,
            t_pool=t_pool,
            KG_d1_feat=KG_d1_feat,
            KG_d2_feat=KG_d2_feat,
            DDI_type_index=DDI_type_index
        )
        return moe_out


    def do_compute(batch, device, model, model_mol, model_kg):
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch

        pos_out = _forward_single_branch(pos_tri, model, model_mol, model_kg)
        probas_pred.append(torch.sigmoid(pos_out["fuse_score"].detach()).cpu())
        ground_truth.append(np.ones(len(pos_out["fuse_score"])))

        neg_out = _forward_single_branch(neg_tri, model, model_mol, model_kg)
        probas_pred.append(torch.sigmoid(neg_out["fuse_score"].detach()).cpu())
        ground_truth.append(np.zeros(len(neg_out["fuse_score"])))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        return pos_out, neg_out, probas_pred, ground_truth


    ######################### Metrics ######################
    def do_compute_metrics(probas_pred, target):
        pred = (probas_pred >= 0.5).astype(int)
        acc = metrics.accuracy_score(target, pred)
        auroc = metrics.roc_auc_score(target, probas_pred)
        f1_score = metrics.f1_score(target, pred)
        precision = metrics.precision_score(target, pred)
        recall = metrics.recall_score(target, pred)
        p, r, t = metrics.precision_recall_curve(target, probas_pred)
        int_ap = metrics.auc(r, p)
        ap = metrics.average_precision_score(target, probas_pred)

        return acc, auroc, f1_score, precision, recall, int_ap, ap


    ######################### Train / Eval ######################
    def train(model, model_mol, model_kg, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
        max_acc = 0.0
        pareto_solver = ParetoMGDASolver(max_iter=args.pareto_iters)

        print('Starting training at', datetime.today())

        for i in range(1, n_epochs + 1):
            start = time.time()

            model.train()
            model_mol.eval()
            model_kg.eval()

            train_loss = 0.0
            train_probas_pred = []
            train_ground_truth = []

            alpha_hist = []

            for batch in tqdm(train_data_loader, total=len(train_data_loader)):
                pos_out, neg_out, probas_pred, ground_truth = do_compute(batch, device, model, model_mol, model_kg)
                train_probas_pred.append(probas_pred)
                train_ground_truth.append(ground_truth)

                # Three Pareto objectives
                loss_fuse, _, _ = loss_fn(pos_out["fuse_score"], neg_out["fuse_score"])
                loss_mol, _, _ = loss_fn(pos_out["mol_score_routed"], neg_out["mol_score_routed"])
                loss_kg, _, _ = loss_fn(pos_out["kg_score_routed"], neg_out["kg_score_routed"])

                bal_loss = 0.0
                if pos_out["bal_loss"] is not None and neg_out["bal_loss"] is not None:
                    bal_loss = 0.5 * (pos_out["bal_loss"] + neg_out["bal_loss"])

                pareto_losses = [loss_fuse, loss_mol, loss_kg]
                alpha = pareto_solver.solve(pareto_losses, model)
                alpha_hist.append(alpha.cpu().numpy())

                total_loss = alpha[0] * loss_fuse + alpha[1] * loss_mol + alpha[2] * loss_kg + bal_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item() * len(pos_out["fuse_score"])

            train_loss /= len(train_data)

            with torch.no_grad():
                train_probas_pred = np.concatenate(train_probas_pred)
                train_ground_truth = np.concatenate(train_ground_truth)
                train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = \
                    do_compute_metrics(train_probas_pred, train_ground_truth)

                model.eval()
                val_loss = 0.0
                val_probas_pred = []
                val_ground_truth = []

                for batch in tqdm(val_data_loader, total=len(val_data_loader)):
                    pos_out, neg_out, probas_pred, ground_truth = do_compute(batch, device, model, model_mol, model_kg)

                    val_probas_pred.append(probas_pred)
                    val_ground_truth.append(ground_truth)

                    loss_fuse, _, _ = loss_fn(pos_out["fuse_score"], neg_out["fuse_score"])
                    val_loss += loss_fuse.item() * len(pos_out["fuse_score"])

                val_loss /= len(val_data)
                val_probas_pred = np.concatenate(val_probas_pred)
                val_ground_truth = np.concatenate(val_ground_truth)

                val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = \
                    do_compute_metrics(val_probas_pred, val_ground_truth)

                if val_acc > max_acc:
                    max_acc = val_acc
                    torch.save(model.state_dict(), pkl_name)

            if scheduler is not None:
                scheduler.step()

            alpha_hist = np.array(alpha_hist)
            mean_alpha = alpha_hist.mean(axis=0)

            print(
                f'Epoch: {i} ({time.time() - start:.4f}s), '
                f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, '
                f'pareto_alpha=[fuse:{mean_alpha[0]:.4f}, mol:{mean_alpha[1]:.4f}, kg:{mean_alpha[2]:.4f}]'
            )
            print(
                f'\t\tval_acc: {val_acc:.4f}, val_auc_roc: {val_auc_roc:.4f}, '
                f'val_f1: {val_f1:.4f}, val_precision: {val_precision:.4f}'
            )
            print(
                f'\t\tval_recall: {val_recall:.4f}, val_int_ap: {val_int_ap:.4f}, val_ap: {val_ap:.4f}'
            )


    def test(test_data_loader, model, model_mol, model_kg):
        test_probas_pred = []
        test_ground_truth = []

        model.eval()
        model_mol.eval()
        model_kg.eval()

        with torch.no_grad():
            for batch in tqdm(test_data_loader, total=len(test_data_loader)):
                pos_out, neg_out, probas_pred, ground_truth = do_compute(batch, device, model, model_mol, model_kg)
                test_probas_pred.append(probas_pred)
                test_ground_truth.append(ground_truth)

            test_probas_pred = np.concatenate(test_probas_pred)
            test_ground_truth = np.concatenate(test_ground_truth)

            test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap = \
                do_compute_metrics(test_probas_pred, test_ground_truth)

        print('\n')
        print('============================== Test Result ==============================')
        print(
            f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f}, '
            f'test_f1: {test_f1:.4f}, test_precision:{test_precision:.4f}'
        )
        print(
            f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f}, test_ap: {test_ap:.4f}'
        )


    ######################### Build Models ######################
    model = Moe.MVN_DDI(
        hidden_dim=args.hidden_dim,
        ddi_type=args.ddi_types,
        drop=0.2,
        bal_beta=args.bal_beta,
        moe_topk=args.moe_topk
    ).to(device=device)

    loss = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

    model_mol = Str_expert.MVN_DDI(
        [n_atom_feats, 2048, 200], 17, kge_dim, kge_dim,
        rel_total, [64, 64, 64, 64], [2, 2, 2, 2], 64, 0.0
    )
    model_mol.load_state_dict(torch.load(pkl_mol, map_location=device))
    model_mol.to(device=device)

    model_kg = KG_expert(hidden_dim=args.hidden_dim, hyperNum=args.hyper_num, ddi_type=args.ddi_types)
    model_kg.load_state_dict(torch.load(pkl_kg, map_location=device))
    model_kg.to(device=device)

    # Freeze both pretrained experts
    for p in model_mol.parameters():
        p.requires_grad = False
    for p in model_kg.parameters():
        p.requires_grad = False

    train(
        model, model_mol, model_kg,
        train_data_loader, test_data_loader,
        loss, optimizer, n_epochs, device, scheduler
    )

    model.load_state_dict(torch.load(pkl_name, map_location=device))
    model.to(device=device)
    test(test_data_loader, model, model_mol, model_kg)