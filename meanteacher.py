import tqdm
import time
import collections

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

from model_factory import get_model
from ema import ExponentialMovingAverage

from training_utils import AverageMeter, consistency_rampup_weight, accuracy
  
class Trainer:
    def __init__(self, cfg, 
        model,
        ema_model,
        optimizer):

        self.cfg = cfg
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer

        self.sup_criterion = nn.CrossEntropyLoss(ignore_index=self.cfg.UNLABLED) # mark -1 as unlabeled data
        self.unp_criterion = nn.MSELoss()

        self._create_ema_updater()
        self._create_summary_writer()

    def _create_summary_writer(self):
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.writer = SummaryWriter(f'runs/meanteacher/time_str')

    def train(self, epoch, data_loader):
        self.model.train()
        self.ema_model.train()

        pbar = tqdm(data_loader)
        meters = collections.defaultdict(AverageMeter())

        for i, (data, ema_data, targets) in enumerate(pbar):
            pbar.set_description(f"Train Epoch: {epoch} [{i}/{len(pbar)}]")

            sdata, tdata = data.to(self.cfg.device), ema_data.to(self.cfg.device)
            targets = targets.to(self.cfg.device)
            labeled_batch_size = targets.ne(-1).sum()

            # Train student on batch1
            spreds = self.model(sdata)
            
            # Train teacher on batch2
            with torch.no_grad():
                tpreds = self.model(tdata)
                tpreds.requires_grad = False # disable gradient

            # compute supervised loss for student predictions
            spreds = F.softmax(spreds)
            student_sup_loss = self.sup_loss(spreds, targets) 
            meters["student_sup_loss"].update(student_sup_loss.item())

            # compute supervised loss for teacher predictions
            tpreds = F.softmax(tpreds)
            teacher_sup_loss = self.sup_loss(tpreds, targets) 
            meters["teacher_sup_loss"].update(teacher_sup_loss.item())

            # compute unsupervised loss between teacher and student predictions
            unp_loss = self.unp_loss(spreds, tpreds) / self.cfg.num_classes
            meters["unp_loss"].update(unp_loss.item())

            # main objective
            alpha = consistency_rampup_weight(self.cfg.unp_weight, epoch, self.cfg.rampup_length)
            loss = student_sup_loss + alpha*unp_loss
            meters["loss"].update(loss.item())
            meters["consistency_weight"].update(alpha)

            # compute accuracy and error for student model
            student_top1_acc, student_top5_acc = accuracy(spreds, targets, topk=(1,5)) # students
            meters["student_top1_acc"].update(student_top1_acc, labeled_batch_size)
            meters["student_top5_acc"].update(student_top5_acc, labeled_batch_size)
            meters["student_top1_error"].update(100.-student_top1_acc, labeled_batch_size)
            meters["student_top5_error"].update(100.-student_top5_acc, labeled_batch_size)

            # compute accuracy and error for teacher model
            teacher_top1_acc, teacher_top5_acc = accuracy(tpreds, targets, topk=(1,5)) # teachers
            meters["teacher_top1_acc"].update(teacher_top1_acc, labeled_batch_size)
            meters["teacher_top5_acc"].update(teacher_top5_acc, labeled_batch_size)
            meters["teacher_top1_error"].update(100.-teacher_top1_acc, labeled_batch_size)
            meters["teacher_top5_error"].update(100.-teacher_top5_acc, labeled_batch_size)

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() # update weights
            
            # perform EMA on weigths of student model
            self.ema_updater.update(self.model.parameters())
            self.ema_updater.copy_to(self.ema_model.parameters())

            # print log
            if i % self.cfg.print_interval == 0:
                pbar.set_postfix(
                    student_sup_loss=student_sup_loss.item(),
                    teacher_sup_loss=teacher_sup_loss.item(),
                    unp_loss=unp_loss.item(),
                    consistency_weight=alpha,
                    loss=loss.item(),
                    student_top1_acc=student_top1_acc,
                    student_top5_acc=student_top5_acc,
                    teacher_top1_acc=teacher_top1_acc,
                    teacher_top5_acc=teacher_top5_acc,
                )

        # logging
        for k, v in meters.items():
            if "top" in k: # train_acc
                self.writer.add_scalar("train_acc/{k}", v.avg)
            else: # train_loss
                self.writer.add_scalar("train_loss/{k}", v.avg)

    def val(self, epoch, data_loader):
        """
        Validate model on validation set 

        Returns:
        - student_top1_acc, teacher_top1_acc
        """
        self.model.eval()
        self.ema_model.eval()

        pbar = tqdm(data_loader)
        meters = collections.defaultdict(AverageMeter())

        for i, (data, targets) in enumerate(pbar):
            pbar.set_description(f"Val Epoch: {epoch} [{i}/{len(pbar)}]")

            data = data.to(self.cfg.device)
            targets = targets.to(self.cfg.device)
            labeled_batch_size = targets.ne(self.cfg.UNLABELED).sum()

            # validate student on data
            spreds = self.model(data)
            
            # validate teacher on data
            tpreds = self.model(data)

            # compute supervised loss for student predictions
            spreds = F.softmax(spreds)
            student_sup_loss = self.sup_loss(spreds, targets) 
            meters["student_sup_loss"].update(student_sup_loss.item())

            # compute supervised loss for teacher predictions
            tpreds = F.softmax(tpreds)
            teacher_sup_loss = self.sup_loss(tpreds, targets) 
            meters["teacher_sup_loss"].update(teacher_sup_loss.item())

            # compute accuracy and error for student model
            student_top1_acc, student_top5_acc = accuracy(spreds, targets, topk=(1,5)) # students
            meters["student_top1_acc"].update(student_top1_acc, labeled_batch_size)
            meters["student_top5_acc"].update(student_top5_acc, labeled_batch_size)
            meters["student_top1_error"].update(100.-student_top1_acc, labeled_batch_size)
            meters["student_top5_error"].update(100.-student_top5_acc, labeled_batch_size)

            # compute accuracy and error for teacher model
            teacher_top1_acc, teacher_top5_acc = accuracy(tpreds, targets, topk=(1,5)) # teachers
            meters["teacher_top1_acc"].update(teacher_top1_acc, labeled_batch_size)
            meters["teacher_top5_acc"].update(teacher_top5_acc, labeled_batch_size)
            meters["teacher_top1_error"].update(100.-teacher_top1_acc, labeled_batch_size)
            meters["teacher_top5_error"].update(100.-teacher_top5_acc, labeled_batch_size)

            # print log
            if i % self.cfg.print_interval == 0:
                pbar.set_postfix(
                    student_sup_loss=student_sup_loss.item(),
                    teacher_sup_loss=teacher_sup_loss.item(),
                    student_top1_acc=student_top1_acc,
                    student_top5_acc=student_top5_acc,
                    teacher_top1_acc=teacher_top1_acc,
                    teacher_top5_acc=teacher_top5_acc,
                )

        # logging
        for k, v in meters.items():
            if "top" in k: # train_acc
                self.writer.add_scalar("val_acc/{k}", v.avg)
            else: # train_loss
                self.writer.add_scalar("val_loss/{k}", v.avg)

        return meters["student_top1_acc"].avg, meters["teacher_top1_acc"].avg

    def _set_device(self):
        if len(self.cfg.device_ids) > 1:
            self.model = nn.DataParallel(
                self.model, device_ids=self.cfg.device_ids,
                chunk_sizes=self.chunk_sizes).to(self.cfg.device)

            self.ema_model = nn.DataParallel(
                self.ema_model, device_ids=self.cfg.device_ids,
                chunk_sizes=self.cfg.chunk_sizes).to(self.cfg.device)

        else:
            self.model = self.model.to(self.cfg.device)
            self.ema_model = self.ema_model.to(self.cfg.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.cfg.device, non_blocking=True)
            
    def _create_ema_updater(self):
        self.ema_updater = ExponentialMovingAverage(self.model.parameters(), self.cfg.ema_decay, 
                use_num_updates=self.cfg.use_num_updates, rampup_steps=self.cfg.rampup_steps,
                rampup_decay=self.cfg.rampup_decay)
        self.ema_updater.store(self.model.parameters()) # store pretrained weights of model
        # self.ema_updater.copy_to(self.ema_model.parameters()) # load weight for ema_model
        self.ema_updater.set_params(self.model.parameters()) # set initial params
        

if __name__ == "__main__":
    pass
    
    

