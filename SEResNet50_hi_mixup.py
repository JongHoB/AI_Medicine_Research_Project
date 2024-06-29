
###
import timm
import copy
from tqdm import tqdm
from PIL import Image
from fastprogress import master_bar, progress_bar
from torchtoolbox.tools import mixup_data, mixup_criterion
from torch.autograd import Variable
from torchvision import models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import time
import sys
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
import warnings
warnings.filterwarnings("ignore")
####
####
##


if __name__ == '__main__':

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    test_scores_list = []

    cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
            'Fracture', 'Support Devices']

    class ChestDataset(Dataset):
        def __init__(self, csv_file, dir, transform=None):
            df = pd.read_csv(csv_file)
            df.fillna(0, inplace=True)  # Nan 0으로 간주
            for j in range(len(df)):
                for i in range(5, 19):
                    if df.iloc[j, i] < 0:
                        df.iloc[j, i] = 0.5    # -1 - 0.5으로 간주
                        
                    if i == 7 and df.iloc[j, i] == 1: # Cardiomegaly가 1이면 Enlarged Cardiomediastinum 1로 변경
                        df.iloc[j,6]=1
                    
                    if i == 9 or i == 10 or i== 11 or i == 12 or i== 13 : # Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis가 1이면 Lung Opacity 1로 변경
                        if df.iloc[j, i] == 1:
                            df.iloc[j, 8] = 1
    
                    if i==12 and df.iloc[j,i]==1: # Pneumonia가 1이면 Consolidation 1로 변경
                        df.iloc[j,11]=1
                    

            self.data = df
            self.dir = dir
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image_path = self.dir+self.data.iloc[idx, 0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            y = self.data.iloc[idx, 5:19]
            y = torch.tensor(y, dtype=torch.float32)
            return image, y

    class ChestTest(Dataset):
        def __init__(self, csv_file, dir, transform=None):
            df = pd.read_csv(csv_file)

            self.data = df
            self.dir = dir
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image_path = self.dir+self.data.iloc[idx, 0]
            image = Image.open(image_path)
            image = self.transform(image)
            y = self.data.iloc[idx, 1:]
            y = torch.tensor(y, dtype=torch.float32)
            return image, y

    class ResNet50(nn.Module):
        def __init__(self, num_classes=14, is_trained=False):

            super().__init__()
            self.net = timm.create_model('seresnet50', pretrained=True, num_classes=14)

        def forward(self, inputs):
            return self.net(inputs)
    def mixup_data(x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_model(model, train_loader, epoch, optimizer, criterion, scheduler, device, BATCH_SIZE,mixup_labels=[7, 12], mixup_alpha=1.0):
        model.train()

        total_loss = 0
        epoch_start_time = time.time()  # epoch 시작 시간
        total_correct = 0
        total_items = 0
        print("epoch: {}".format(epoch+1))
        train_progress_bar = tqdm(
            train_loader, desc="Training", total=len(train_loader))
        for data, labels in train_progress_bar:
            total_items += data.size(0)
            d, l = data.to(device), labels.to(device)
            optimizer.zero_grad()
            result_hat = model(d)
            loss = criterion(result_hat, l)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_correct += ((result_hat > 0.7) == l).sum().item()
            train_progress_bar.set_postfix(
                {'Epoch': epoch+1, 'items': total_items, 'Loss': "{:.4f}".format(loss.item())})
        train_loss = total_loss / len(train_loader)
        # train_loss_list.append(train_loss)

        accuracy = total_correct / (total_items * 14) * 100
        # train_acc_list.append(accuracy)

        epoch_end_time = time.time()
        print('Epoch: {} Total Items: {} Training Loss: {:.4f} Accuracy: {:.4f} Time:{:.2f}s'.format(
            epoch + 1, total_items, train_loss, accuracy, epoch_end_time-epoch_start_time))

        # save the print to a file
        with open('SEResNet50_hierarchy_Train_Log.txt', 'a') as f:
            print('Epoch: {} Total Items: {} Training Loss: {:.4f} Accuracy: {:.4f} Time:{:.2f}s'.format(
                epoch + 1, total_items, train_loss, accuracy, epoch_end_time-epoch_start_time), file=f)
            
    def train_model_mixup(model, train_loader, epoch, optimizer, criterion, scheduler, device, BATCH_SIZE,alpha=1.0):
        model.train()

        total_loss = 0
        epoch_start_time = time.time()  # epoch 시작 시간
        total_correct = 0
        total_items = 0
        print("epoch: {}".format(epoch+1))
        train_progress_bar = tqdm(
            train_loader, desc="Training Mixup", total=len(train_loader))
        for data, labels in train_progress_bar:

            # accuracy 계산을 위해 전체 item 수 계산
            total_items += data.size(0)

            d, l = data.to(device), labels.to(device)
            d, targets_a, targets_b, lam = mixup_data(d, l, alpha)

            optimizer.zero_grad()

            result_hat = model(d)
            loss = mixup_criterion(criterion, result_hat, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            # Multi label 각 label이 0.7 이상이면 1로 간주, 0.7 미만이면 0으로 간주
            # labels 와 비교
            total_correct += ((result_hat > 0.7) == l).sum().item()

            train_progress_bar.set_postfix(
                {'Epoch': epoch+1, 'items': total_items, 'Loss': "{:.4f}".format(loss.item())})
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)

        accuracy = total_correct / (total_items * 14) * 100
        train_acc_list.append(accuracy)

        epoch_end_time = time.time()
        print('Epoch: {} Total Items: {} Training Mixup Loss: {:.4f} Accuracy: {:.4f} Time:{:.2f}s'.format(
            epoch + 1, total_items, train_loss, accuracy, epoch_end_time-epoch_start_time))

        # save the print to a file
        with open('SEResNet50_hierarchy_Train_Log.txt', 'a') as f:
            print('Epoch: {} Total Items: {} Training Mixup Loss: {:.4f} Accuracy: {:.4f} Time:{:.2f}s'.format(
                epoch + 1, total_items, train_loss, accuracy, epoch_end_time-epoch_start_time), file=f)

    def valid_model(model, val_loader, epoch, criterion, device, batch_size, scheduler):
        model.eval()
        val_loss = 0
        total_items = 0
        with torch.no_grad():
            accuracy = 0
            val_progress_bar = tqdm(
                val_loader, desc="Validation", total=len(val_loader))
            for val_data, val_labels in val_progress_bar:
                total_items += val_data.size(0)
                vd, vl = val_data.to(device), val_labels.to(device)
                val_result_hat = model(vd)
                val_loss += criterion(val_result_hat, vl).item()
                correct_pred = (val_result_hat > 0.7).float() == vl
                accuracy += correct_pred.sum().item()

                val_progress_bar.set_postfix(
                    {'Epoch': epoch+1, 'items': total_items, 'Loss': "{:.4f}".format(val_loss)})

        val_loss = val_loss / len(val_loader)
        valid_loss_list.append(val_loss)

        accuracy = accuracy / total_items / 14 * 100
        valid_acc_list.append(accuracy)

        # scheduler.step(val_loss)

        print('Valid Epoch: {} Total Items {} Validation Loss: {:.4f} Accuracy: {:.2f}%'.format(
            epoch + 1, total_items, val_loss, accuracy))

        # save the print to a file

        with open('SEResNet50_hierarchy_Valid_Log.txt', 'a') as f:
            print('Valid Epoch: {} Total Items {} Validation Loss: {:.4f} Accuracy: {:.2f}%'.format(
                epoch + 1, total_items, val_loss, accuracy), file=f)

        return val_loss

    def loss_plot(train_loss_list, valid_loss_list):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_list, label='train_loss')
        plt.plot(valid_loss_list, label='valid_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # save the plot
        plt.savefig('SEResNet50_hierarchy_loss_plot.png')

    def acc_plot(train_acc_list, valid_acc_list):
        plt.figure(figsize=(10, 5))
        plt.plot(train_acc_list, label='train_acc')
        plt.plot(valid_acc_list, label='valid_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # save the plot
        plt.savefig('SEResNet50_hierarchy_acc_plot.png')

    # in test_model(), use test_set to test the model and get accuracy, auc-roc graph, precision, recall, f1 score for each label (this is for multi-label classification)

    @torch.no_grad()
    def test_model(model, test_loader, criterion, device):
        model.eval()
        all_preds = []
        all_labels = []

        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        test_loss = criterion(all_preds, all_labels).item()

        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

        # calculate accuracy per label
        # calculate auc_roc per label
        # calculate precision, recall, f1 score per label
        scores = []
        metrics=[]
        for i in range(14):
            # count true positive, true negative, false positive, false negative
            metrics.append([np.sum((np.array(all_labels[:, i])==1) & (np.array(all_preds[:, i])>0.7)),
                            np.sum((np.array(all_labels[:, i])==0) & (np.array(all_preds[:, i])<0.7)),
                            np.sum((np.array(all_labels[:, i])==0) & (np.array(all_preds[:, i])>0.7)),
                            np.sum((np.array(all_labels[:, i])==1) & (np.array(all_preds[:, i])<0.7))])
            
            accuracy = accuracy_score(all_labels[:, i], all_preds[:, i] > 0.7)
            auc_roc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            precision = precision_score(
                all_labels[:, i], all_preds[:, i] > 0.7)
            recall = recall_score(all_labels[:, i], all_preds[:, i] > 0.7)
            f1 = f1_score(all_labels[:, i], all_preds[:, i] > 0.7)
            scores.append([accuracy, auc_roc, precision, recall, f1])

        scores = pd.DataFrame(
            scores, columns=['Accuracy', 'AUC-ROC', 'Precision', 'Recall', 'F1 Score'])
        test_scores_list.append(scores)
        print(scores)
        
        metrics=pd.DataFrame(metrics, columns=['TP', 'TN', 'FP', 'FN'])
        print(metrics)

        # save the print to a file

        with open('SEResNet50_hierarchy_Test_Log.txt', 'a') as f:
            print(scores, file=f)
            print(metrics, file=f)

        print('Test Loss: {:.4f}'.format(test_loss))

        # save the print to a file

        with open('SEResNet50_hierarchy_Test_Log.txt', 'a') as f:
            print('Test Loss: {:.4f}'.format(test_loss), file=f)

        # plot auc_roc curve for each label
        for i in range(14):
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Label {}'.format(cols[i]))
            plt.savefig('SEResNet50_hierarchy_ROC_Curve_Label_{}.png'.format(cols[i]))

    # in scores_plot(), use test_scores_list to plot the accuracy, auc-roc, precision, recall, f1 score for each label

    def scores_plot(test_scores_list, epoch):
        for i in range(14):
            accuracy = [test_scores_list[j].iloc[i, 0]
                        for j in range(len(test_scores_list))]
            auc_roc = [test_scores_list[j].iloc[i, 1]
                       for j in range(len(test_scores_list))]
            precision = [test_scores_list[j].iloc[i, 2]
                         for j in range(len(test_scores_list))]
            recall = [test_scores_list[j].iloc[i, 3]
                      for j in range(len(test_scores_list))]
            f1 = [test_scores_list[j].iloc[i, 4]
                  for j in range(len(test_scores_list))]

            plt.figure(figsize=(10, 5))
            plt.plot(accuracy, label='Accuracy', linestyle='dashed')
            plt.plot(auc_roc, label='AUC-ROC', linestyle='dashed')
            plt.plot(precision, label='Precision', linestyle='dashed')
            plt.plot(recall, label='Recall', linestyle='dashed')
            plt.plot(f1, label='F1 Score', linestyle='dashed')
            plt.xlabel('Epochs')
            plt.ylabel('Scores')
            plt.title('Epoch:{}'.format(epoch+1)+' Label:{}'.format(cols[i]))
            plt.legend()
            plt.savefig('SEResNet50_hierarchy_Scores_Label_{}.png'.format(cols[i]))

    # use 2 gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    print(device)

    base_path = os.getcwd()+'/'
    print(base_path)

    BATCH_SIZE = 64
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.472, 0.472, 0.472], [0.320, 0.320, 0.320]),])

    train_tranform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAdjustSharpness(4),
        transforms.ToTensor(),
        transforms.Normalize([0.472, 0.472, 0.472], [0.320, 0.320, 0.320]),])

    train_data = ChestDataset(
        base_path+'CheXpert-v1.0/train.csv', base_path, transform=train_tranform)

    valid_data = ChestDataset(
        base_path+'CheXpert-v1.0/valid.csv', base_path, transform=valid_transform)
    test_data = ChestTest(base_path+'test_labels.csv',
                          base_path, transform=valid_transform)
    print("data")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    print("load")

    model = ResNet50()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        # model=nn.DataParallel(model)
    model.to(device)

    best_model = copy.deepcopy(model.state_dict())

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300, eta_min=0.0001)

    print("train start")
    # best_acc = 0
    # accuracy = 0
    best_loss = 100
    loss = 0
    for t in range(300):
        train_model(model, train_loader, t, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, device=device, BATCH_SIZE=BATCH_SIZE)
        train_model_mixup(model, train_loader, t, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, device=device, BATCH_SIZE=BATCH_SIZE,alpha=1.0)
        loss = valid_model(model, valid_loader, t,
                           criterion, device, BATCH_SIZE, scheduler)
        torch.save(model.state_dict(), base_path+'SEResNet50_hierarchy.pth')

        loss_plot(train_loss_list, valid_loss_list)
        acc_plot(train_acc_list, valid_acc_list)

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, base_path+'SEResNet50_hierarchy_best.pth')
            test_model(model, test_loader, criterion, device)
            scores_plot(test_scores_list, t)
