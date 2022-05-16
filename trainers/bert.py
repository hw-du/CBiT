from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


import torch
import torch.nn as nn


class NTXENTloss(nn.Module):

    def __init__(self, args, device,temperature=1.):
        super(NTXENTloss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device
        self.w1 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.projection_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.w2 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False).to(self.device)
        #self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))
    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, h1, h2,calcsim='dot'):
        b = h1.shape[0]
        if self.args.projectionhead:
            z1, z2 = self.project(h1.view(b*self.args.bert_max_len,self.args.bert_hidden_units)), self.project(h2.view(b*self.args.bert_max_len,self.args.bert_hidden_units))
        else:
            z1, z2 = h1, h2
        z1 = z1.view(b, self.args.bert_max_len*self.args.bert_hidden_units)
        z2 = z2.view(b, self.args.bert_max_len*self.args.bert_hidden_units)
        if calcsim=='dot':
            sim11 = torch.matmul(z1, z1.T) / self.temperature
            sim22 = torch.matmul(z2, z2.T) / self.temperature
            sim12 = torch.matmul(z1, z2.T) / self.temperature
        elif calcsim=='cosine':
            sim11 = self.cosinesim(z1, z1) / self.temperature
            sim22 = self.cosinesim(z2, z2) / self.temperature
            sim12 = self.cosinesim(z1, z2) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        ntxentloss = self.criterion(raw_scores, targets)
        return ntxentloss


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader,  export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.alpha = args.alpha
        self.NTXENTloss = NTXENTloss(args,self.device,self.args.tau)

        self.lambda_=args.lambda_
        self.theta = 0
    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def one_pair_contrastive_learning(self, inputs, calcsim='dot'):
        cl_batch = torch.cat(inputs, dim=0)
        _,cl_sequence_output = self.model(cl_batch)

        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)

        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)

        cl_loss = self.NTXENTloss(cl_output_slice[0],
                                    cl_output_slice[1],calcsim=calcsim)
        return cl_loss


    def calculate_loss(self, batch, enbale_DA=False):
        num_positive=len(batch)//2
        '''
        logits,c_i = self.model(seqs)  # B x T x V
        #c_i2, _,_, logits2 = self.model([seqs, negs])
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        
        loss = self.ce(logits, labels)
        '''
        #logits2 = logits2.view(-1, logits2.size(-1))
        #loss_2=self.ce(logits2, labels)

        if enbale_DA:
            cl_loss = self.one_pair_contrastive_learning([aug_1,aug_2],calcsim=self.args.calcsim)
            #total_loss = loss + self.alpha * cl_loss
            proportion = self.alpha * loss.detach().data.item()/(loss.detach().data.item()+cl_loss.detach().data.item())
            total_loss = loss + proportion * cl_loss
        else:
            pairs = []
            main_loss = 0
            cl_loss=0
            for i in range(num_positive):
                seqs=batch[2*i]
                labels=batch[2*i+1]
                logits_k,c_i_k = self.model(seqs)
                loss_k = self.ce(logits_k.view(-1, logits_k.size(-1)), labels.view(-1))
                main_loss = main_loss+loss_k
                pairs.append(c_i_k)
            for j in range(num_positive):
                for k in range(num_positive):
                    if j!=k:
                        cl_loss = self.NTXENTloss(pairs[j], pairs[k], calcsim=self.args.calcsim) + cl_loss
            num_main_loss = main_loss.detach().data.item()
            num_cl_loss = cl_loss.detach().data.item()
            theta_hat = num_main_loss/(num_main_loss+self.lambda_*num_cl_loss)
            self.theta = self.alpha*theta_hat+(1-self.alpha)*self.theta
            total_loss = main_loss + self.theta*cl_loss

        #cl_loss = self.ContrastiveLossFunction(c_i, h_il, h_jl)
        #total_loss = loss + loss_2 + 0.5*cl_loss
        #total_loss = loss + self.alpha * cl_loss
        #return total_loss,cl_loss
        return total_loss,cl_loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores,_ = self.model(seqs)
        scores = scores[:, -1, :]
        scores[:,0] = -999.999
        scores[:,-1] = -999.999# pad token and mask token should not appear in the logits output

        scores = scores.gather(1, candidates)#the whole item set 

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        return metrics

