import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import copy
import logging
import numpy as np
from tqdm import tqdm
import math

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from models.network import MANet
from models.decomposed_lora import Attention_LoRA
# from models.decomposed_lora_kv import Attention_LoRA

from models.functions import mixup, cutmix
from utils.schedulers import CosineSchedule
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import count_parameters
from models.losses import AngularPenaltySMLoss
import re

class Learner(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = MANet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module._init_params(args)
            
        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.scale = args["scale"]
        self.margin = args["margin"]
        self.total_sessions = args["total_sessions"]
        self.total_classnum = self.args["increment"] * self.total_sessions + self.args["init_cls"]

        self.dataset = args["dataset"]
        self.logit_norm = args["logit_norm"]
        if self.logit_norm == "none":
            self.logit_norm = None
        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.task_sizes = []

        # class prototypes
        self._class_means = None
        self._class_covs = None
        self._old_class_covs = None
        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self._old_class_covs = None

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)
        
        # Semantic Shift old embedding 
        if self._cur_task > 0 and self._old_network is not None:
            self._old_network.to(self._device)

        self._train(self.train_loader, self.test_loader)

        # update mean and cov and classifier alignment
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task > 0 and self.args['ca'] is True:
            self._stage2_compact_classifier(self.task_sizes[-1])

        # if self.args["use_RP"]:
        #     print("DLORA with random projection and analytic classifier")
        #     for _, p in self._network.named_parameters():
        #         p.requires_grad = False
        #     if self._cur_task == 0:
        #         self.setup_RP()
            
        #     train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        #     self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self.batch_size, 
        #                                            shuffle=True, num_workers=self.num_workers)
        #     self.replace_fc(self.train_loader_for_CPs)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.before_task(task=self._cur_task)
                module.cur_matrix.zero_(); module.n_cur_matrix = 0

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

        if len(self._multiple_gpus) > 1:
            self._network = torch.nn.DataParallel(self._network, self._multiple_gpus)

        kk = 0
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                print(f'********** LoRA weights initialization for layer {kk} **********')
                module._init_lora_weight(task=self._cur_task, layer_idx=kk)
                module.set_task_and_stage(task=self._cur_task, layer_idx=kk)
                module.cur_matrix.zero_(); module.n_cur_matrix = 0
                kk += 1

        ############################## set learning rates ##################################
        ############################## set learning rates ##################################
        ############################## set learning rates ##################################
        flora_params, other_params = [], []
        for name, p in self._network.named_parameters():
            if p.requires_grad:
                if 'lora' in name.lower():
                    flora_params.append(p)
                else:
                    other_params.append(p)
        print(f"[Param Group] LoRA params: {len(flora_params)}, Other params: {len(other_params)}")

        enabled = {name for name, p in self._network.named_parameters() if p.requires_grad}
        print(f"[LoRA-Stage] Parameters to be updated: {enabled}")

        lr = self.init_lr if self._cur_task == 0 else self.lrate
        weight_decay = self.init_weight_decay if self._cur_task == 0 else self.weight_decay
        param_groups = [
            {'params': flora_params, 'lr': lr, 'momentum': 0.9, 'weight_decay': weight_decay},
            {'params': other_params, 'lr': lr, 'momentum': 0.9, 'weight_decay': weight_decay}
        ]
        ############################## set learning rates ##################################
        ############################## set learning rates ##################################
        ############################## set learning rates ##################################
                
        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=param_groups)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=param_groups,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=param_groups)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=param_groups,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.cur_matrix.zero_(); module.n_cur_matrix = 0
        
        with torch.no_grad():

            print('*'*10+'Extrace features for merging shared component!'+'*'*10)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    module.after_task(task=self._cur_task)
                    module._process_feature_mat()
                    module.cur_matrix.zero_(); module.n_cur_matrix = 0

        with torch.no_grad():
            print('*'*10+'Extrace features for saving stastistics!'+'*'*10)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    module._process_feature_mat()
                    module.cur_matrix.zero_(); module.n_cur_matrix = 0

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        # logging.info(f"Parameters to be updated: {enabled}")

        prog_bar = tqdm(range(self.run_epoch))
        
        loss_cos:AngularPenaltySMLoss = AngularPenaltySMLoss(loss_type='cosface',s=self.scale,m=self.margin)

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes
                
                ## random partition
                classes:torch.Tensor = torch.unique(targets)
                perm = classes[torch.randperm(len(classes), device=self._device)]
                m = max(1, len(classes)//2)
                g1_classes = perm[:m]; g2_classes = perm[m:]
                g1_idx = torch.isin(targets, g1_classes).nonzero(as_tuple=True)[0]
                g2_idx = torch.isin(targets, g2_classes).nonzero(as_tuple=True)[0]

                ## optimization -- cross-entropy
                if not self.args["use_gao"] or classes.shape[0] == 1 or \
                    (min(g1_idx.shape[0], g2_idx.shape[0]) == 0):
                    output = self._network(inputs)
                    logits = output['logits']
                    loss=loss_cos(logits, targets) 

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                ## optimization -- (meta-training)
                else:
                    (x1, y1) = inputs[g1_idx], targets[g1_idx]; (x2, y2) = inputs[g2_idx], targets[g2_idx]
                    params = [p for name, p in self._network.named_parameters()
                              if p.requires_grad and any(s in name.lower() for s in ["lora"]) ]                    
                    rho = -float(self.args["rho"]) * torch.rand(1).to(self._device)

                    def perturb_and_train(x_src, y_src, x_tgt, y_tgt, coeff:float=1.0):
                        
                        meta_logits:torch.Tensor = self._network(x_src)["logits"]
                        meta_grads = torch.autograd.grad(
                                     loss_cos(meta_logits, y_src), 
                                     params, create_graph=False, retain_graph=False)

                        with torch.no_grad():
                            flat_norm = [g.norm() + 1e-12 for g in meta_grads]
                            eps_list = [rho * g / flat_norm[id] for id, g in enumerate(meta_grads)]
                            for p, e in zip(params, eps_list):
                                p.add_(e)

                        out_t = self._network(x_tgt)["logits"]
                        loss_t = coeff * loss_cos(out_t, y_tgt)
                        optimizer.zero_grad()
                        loss_t.backward()
                        optimizer.step()

                        with torch.no_grad():
                            for p, e in zip(params, eps_list):
                                p.sub_(e)

                        return loss_t, out_t

                    loss2, out2 = perturb_and_train(x1, y1, x2, y2)
                    loss1b, out1b = perturb_and_train(x2, y2, x1, y1)
                    logits = torch.cat([out1b, out2], dim=0); targets = torch.cat([y1, y2], dim=0)
                    loss = loss1b + loss2
                
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, 
                epoch + 1, 
                self.run_epoch, 
                losses / len(train_loader), 
                train_acc
            )
            prog_bar.set_description(info)
        
        # task train finished
        test_acc = self._compute_accuracy(self._network, test_loader)
        final_info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, 
                epoch + 1, 
                self.run_epoch, 
                losses / len(train_loader), 
                train_acc,
                test_acc,
            )
        logging.info(final_info)


    
    def accuracy(self, y_pred, y_true, accuracy_matrix = False):
        assert len(y_pred) == len(y_true), 'Data length error.'
        all_acc = {}
        all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
        
        i = 0
        # Grouped accuracy
        for class_id in range(0, np.max(y_true), self.class_num):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+self.class_num-1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label] 
            i += 1

        # Old accuracy
        idxes = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                            decimals=2)

        # New accuracy
        idxes = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        ret = {}
        # print(len(y_pred), len(y_true))
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                task_id = (targets//self.class_num).cpu()
                y_true_task.append(task_id)
                outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.interface(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for p in self._network.classifier_pool[:self._cur_task+1].parameters():
            p.requires_grad=True
            
        run_epochs = ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
        classifier_lr = self.args["ca_lrate"]
        network_params = [{'params': param_list, 'lr': classifier_lr,
                           'weight_decay': 0.0005}]
        optimizer = optim.SGD(network_params, lr=classifier_lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        # loss_cos:AngularPenaltySMLoss = AngularPenaltySMLoss(loss_type='cosface',s=1.0, m=self.margin)

        self._network.to(self._device)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = self._class_means[c_id].to(self._device)*(0.9+decay)
                cls_cov = self._class_covs[c_id].to(self._device)

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                # -stage two only use classifiers
                outputs = self._network(inp, fc_only=True)
                logits = outputs

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num] * self.args["scale"], tgt)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            # test_acc = self._compute_accuracy(self._network, self.test_loader)
            test_acc = 0.0
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)


    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = torch.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = torch.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = torch.mean(torch.tensor(vectors), dim=0)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean.detach()
            self._class_covs[class_idx, ...] = class_cov.detach()

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
    
    def extract_features(self, trainloader, model, task_id = None):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data, task_id)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list
    
    def _extract_vectors_adv(self, loader, old=False):
        if old:
            network = self._old_network
        else:
            network = self._network
        network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                (_,_inputs, _targets) = batch
                _inputs = _inputs.to(self._device)
                _vectors = network.extract_vector(_inputs)
                vectors.append(_vectors)
                targets.append(_targets)

        return torch.cat(vectors, dim=0), torch.cat(targets, dim=0)


    def shrink_cov(self, cov):
        alpha1 = 10
        alpha2 = 10
        # Compute the mean of the diagonal elements
        diag_mean = torch.mean(torch.diagonal(cov))
        
        # Create a copy of the covariance matrix with zeroed out diagonals
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        
        # Compute the mean of the off-diagonal elements (non-zero entries)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        
        # Identity matrix
        iden = torch.eye(cov.size(0), device=cov.device)
        
        # Shrink the covariance matrix
        cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
        
        return cov_
    
    def _compute_class_invcov(self, data_manager):
        _class_invcovs = torch.zeros((self.class_num, self.feature_dim, self.feature_dim),device=self._device)

        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors_adv(idx_loader, True)

            class_cov = self.shrink_cov(torch.cov(torch.tensor(vectors, dtype=torch.float64).T)) + torch.eye(self.feature_dim).to(self._device) * 1e-3
            _class_invcovs[class_idx-self._known_classes, ...] = torch.linalg.pinv(class_cov).detach()

        return _class_invcovs
    

    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    def setup_RP(self):
        self.initiated_G=False
        self._network.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.weight = torch.nn.Parameter(torch.Tensor(self._total_classes, M).to(device=self._device)) #num classes in task x M
            self._network.W_rand=torch.randn(self._network.dim, M).to(device=self._device)
            self.W_rand=copy.deepcopy(self._network.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.dim #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.use_RP=True
            if self.args['M']>0:
                self._network.W_rand=self.W_rand
            else:
                self._network.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = self._network(data)["features"]
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        # Features_f = F.normalize(torch.cat(Features_f, dim=0), dim=1)
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        def target2onehot(targets, n_classes):
            onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
            onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
            return onehot
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['M']>0:
            Features_h=torch.nn.functional.relu(Features_f @ self._network.W_rand.cpu())
        else:
            Features_h=Features_f
        self.Q=self.Q+Features_h.T @ Y 
        self.G=self.G+Features_h.T @ Features_h
        ridge=self.optimise_ridge_parameter(Features_h,Y)
        Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
        self._network.weight.data=Wo[0:self._total_classes,:].to(device=self._device)

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(3,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
