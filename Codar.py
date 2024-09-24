import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary

from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

class CLSTM(nn.Module):
  def __init__(self, in_channels, hidden_size, num_layers, **kwargs):
    super(CLSTM, self).__init__()
    self.in_channels = in_channels
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.re_LSTM = nn.LSTM(self.in_channels, self.hidden_size, self.num_layers , batch_first = True, bidirectional = True)
    self.im_LSTM = nn.LSTM(self.in_channels, self.hidden_size, self.num_layers, batch_first = True, bidirectional = True)


  def forward(self, x):
        x_re = x.real
        x_im = x.imag

        out_re1, (hn_re1, cn_re1) =  self.re_LSTM(x_re)
        out_re2, (hn_re2, cn_re2) =  self.im_LSTM(x_im)
        out_re = out_re1 - out_re2
        hn_re  = hn_re1  - hn_re2
        cn_re  = cn_re1  - cn_re2

        out_im1, (hn_im1, cn_im1) =  self.re_LSTM(x_re)
        out_im2, (hn_im2, cn_im2) =  self.im_LSTM(x_im)
        out_im = out_im1 + out_im2
        hn_im  = hn_im1  + hn_im2
        cn_im  = cn_im1  + cn_im2

        out = torch.complex(out_re, out_im).to(torch.complex64)
        hn = torch.complex(hn_re, hn_im).to(torch.complex64)
        cn = torch.complex(cn_re, cn_im).to(torch.complex64)

        return out, (hn, cn)

def complex_dropout(input, p=0.5, training=True):
    mask = torch.ones(*input.shape, dtype = torch.float32)
    mask = F.dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    mask = mask.to(input.device)
    return mask*input

class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 32, 5, 1)
        self.bn1 = ComplexBatchNorm2d(32)

        self.conv2 = ComplexConv2d(32, 16, 5, 1)
        self.bn2 = ComplexBatchNorm2d(16)

        self.conv3 = ComplexConv2d(16, 8, 3, 1)
        self.bn3 = ComplexBatchNorm2d(8)

        self.conv4 = ComplexConv2d(8, 16, 3, 1)
        self.bn4 = ComplexBatchNorm2d(16)

        self.flatten = nn.Flatten()

        self.blstm = CLSTM(in_channels=2*3*30, hidden_size=2*3*30, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128*2, 128*2)

        self.fc1 = nn.Linear(48, 4)
        self.fc2 = ComplexLinear(64, 16)
        self.fc3 = ComplexLinear(16, 3)


    def forward(self, x):
        x = x.view(-1,4,2*3*30)
        x,_ = self.blstm(x)
        x = x.view(-1,4,6,60)
        x = x.view(-1, 4, 6, 2, 30)
        x = x.reshape(-1,1, 48,30)
        complex_dropout(x, 0.20)

        x = self.conv1(x)
        x = complex_relu(x)
        x = self.bn1(x)
        x = complex_dropout(x, 0.20)
        # x = complex_max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = complex_relu(x)
        x = self.bn2(x)
        complex_dropout(x, 0.20)
        x = complex_max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = complex_relu(x)
        x = self.bn3(x)
        complex_dropout(x, 0.20)
        x = complex_max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x = complex_relu(x)
        x = self.bn4(x)
        complex_dropout(x, 0.10)
        x = complex_max_pool2d(x, 2, 2)
        x = x.abs()

        x = self.flatten(x)
        return x
class WiFiDataset(Dataset):
   def __init__(self, csi, y):
      self.csi = csi
      self.y = y

   def __len__(self):
      return len(self.y)

   def __getitem__(self, index):
      CSI = self.csi[index]
      Y = self.y[index]
      return CSI, Y


# parameter
batch_size = 16
epoch_num= 100
learning_rate = 0.0001

# # exp1 invasion detection
# class WiFiNet1(nn.Module):
#  def __init__(self):
#   super(WiFiNet1, self).__init__()
#   self.branchA = ComplexNet()
#   self.fc1 = nn.Linear(48, 2)
#
#  def forward(self, x):
#      x = self.branchA(x)
#      x = self.fc1(x)
#      # x = torch.relu(x)
#      # x = self.fc2(x)
#      # x = torch.relu(x)
#      # x = self.fc3(x)
#      x = torch.sigmoid(x)
#      return x
#
# CSI_train = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp1\\CSI_train.mat')['CSI_train']
# CSI_train = torch.from_numpy(np.array(CSI_train, dtype=np.complex64))
# y_train = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp1\\Y_train.mat')['Y_train']
# y_train = torch.from_numpy(np.array(y_train-1, dtype=int))
# # y_train = F.one_hot(y_train.long(),num_classes=3)
# y_train = y_train.long()
# y_train = torch.squeeze(y_train)
#
# CSI_test = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp1\\CSI_test.mat')['CSI_test']
# CSI_test = torch.from_numpy(np.array(CSI_test, dtype=np.complex64))
# y_test = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp1\\Y_test.mat')['Y_test']
# y_test = torch.from_numpy(np.array(y_test-1, dtype=np.int))
# # y_test = F.one_hot(y_test.long(),num_classes=3)
# y_test = y_test.long()
# y_test = torch.squeeze(y_test)
#
# train_set = WiFiDataset(CSI_train, y_train)
# test_set = WiFiDataset(CSI_test, y_test)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#
# net = WiFiNet1().cuda(0)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#
#
# for epoch in range(epoch_num):  # loop over the dataset multiple times
#    train_acc = 0.0
#    train_loss = 0.0
#    test_acc = 0.0
#    test_loss = 0.0
#
#    net.train()
#    for i, (csi, label) in enumerate(train_loader):
#        optimizer.zero_grad()
#        train_pred = net(csi.cuda(0))
#        batch_loss = criterion(train_pred, label.cuda(0))
#        batch_loss.backward()
#        optimizer.step()
#
#        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == label.numpy())
#        train_loss += batch_loss.item()
#
#    net.eval()
#    with torch.no_grad():
#       for i, (csi, label) in enumerate(test_loader):
#          test_pred = net(csi.cuda(0))
#          batch_loss = criterion(test_pred, label.cuda(0))
#
#          test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == label.numpy())
#          test_loss += batch_loss.item()
#       print('epoch %d: train loss: %.4f | test loss %.4f | train acc %.4f | test acc %.4f' %(epoch, train_loss/train_set.__len__(),test_loss/test_set.__len__(),train_acc/train_set.__len__(),test_acc/test_set.__len__()))
#
# # draw confusion matrix
#    net.eval()
#
#    gt_label = np.array([])
#    pred_label = np.array([])
#    with torch.no_grad():
#       for i, (csi,  label) in enumerate(test_loader):
#          test_pred = net(csi.cuda(0))
#          test_pred = torch.argmax(test_pred, dim=1)
#
#
#          gt_label = np.concatenate((gt_label, label.numpy()))
#          pred_label = np.concatenate((pred_label, test_pred.cpu().numpy()))
#       conf_matrix = confusion_matrix(gt_label, pred_label)
#       plt.figure(figsize=(8, 6))
#       sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                   xticklabels=[f'Class {i}' for i in range(2)],
#                   yticklabels=[f'Class {i}' for i in range(2)])
#       plt.xlabel('Predicted Labels')
#       plt.ylabel('True Labels')
#       plt.title('Confusion Matrix')
#       plt.show()

# exp2 invaded floor
# class WiFiNet2(nn.Module):
#  def __init__(self):
#   super(WiFiNet2, self).__init__()
#   self.branchA = ComplexNet()
#   self.fc1 = nn.Linear(48, 4)
#
#  def forward(self, x):
#      x = self.branchA(x)
#      x = self.fc1(x)
#      # x = torch.relu(x)
#      # x = self.fc2(x)
#      # x = torch.relu(x)
#      # x = self.fc3(x)
#      x = torch.sigmoid(x)
#      return x
#
# CSI_train = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp2\\CSI_train.mat')['CSI_train']
# CSI_train = torch.from_numpy(np.array(CSI_train, dtype=np.complex64))
# y_train = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp2\\Y_train.mat')['Y_train']
# y_train = torch.from_numpy(np.array(y_train-1, dtype=int))
# # y_train = F.one_hot(y_train.long(),num_classes=3)
# y_train = y_train.long()
# y_train = torch.squeeze(y_train)
#
# CSI_test = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp2\\CSI_test.mat')['CSI_test']
# CSI_test = torch.from_numpy(np.array(CSI_test, dtype=np.complex64))
# y_test = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp2\\Y_test.mat')['Y_test']
# y_test = torch.from_numpy(np.array(y_test-1, dtype=np.int))
# # y_test = F.one_hot(y_test.long(),num_classes=3)
# y_test = y_test.long()
# y_test = torch.squeeze(y_test)
#
# train_set = WiFiDataset(CSI_train, y_train)
# test_set = WiFiDataset(CSI_test, y_test)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#
# net = WiFiNet2().cuda(0)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#
#
# for epoch in range(epoch_num):  # loop over the dataset multiple times
#    train_acc = 0.0
#    train_loss = 0.0
#    test_acc = 0.0
#    test_loss = 0.0
#
#    net.train()
#    for i, (csi, label) in enumerate(train_loader):
#        optimizer.zero_grad()
#        train_pred = net(csi.cuda(0))
#        batch_loss = criterion(train_pred, label.cuda(0))
#        batch_loss.backward()
#        optimizer.step()
#
#        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == label.numpy())
#        train_loss += batch_loss.item()
#
#    net.eval()
#    with torch.no_grad():
#       for i, (csi, label) in enumerate(test_loader):
#          test_pred = net(csi.cuda(0))
#          batch_loss = criterion(test_pred, label.cuda(0))
#
#          test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == label.numpy())
#          test_loss += batch_loss.item()
#       print('epoch %d: train loss: %.4f | test loss %.4f | train acc %.4f | test acc %.4f' %(epoch, train_loss/train_set.__len__(),test_loss/test_set.__len__(),train_acc/train_set.__len__(),test_acc/test_set.__len__()))
#
# # 画出混淆矩阵
#    net.eval()
#
#    gt_label = np.array([])
#    pred_label = np.array([])
#    with torch.no_grad():
#       for i, (csi,  label) in enumerate(test_loader):
#          test_pred = net(csi.cuda(0))
#          test_pred = torch.argmax(test_pred, dim=1)
#
#
#          gt_label = np.concatenate((gt_label, label.numpy()))
#          pred_label = np.concatenate((pred_label, test_pred.cpu().numpy()))
#       conf_matrix = confusion_matrix(gt_label, pred_label)
#       plt.figure(figsize=(8, 6))
#       sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                   xticklabels=[f'Class {i}' for i in range(4)],
#                   yticklabels=[f'Class {i}' for i in range(4)])
#       plt.xlabel('Predicted Labels')
#       plt.ylabel('True Labels')
#       plt.title('Confusion Matrix')
#       plt.show()

# # exp3 Intruded Floor Identification
class WiFiNet3(nn.Module):
 def __init__(self):
  super(WiFiNet3, self).__init__()
  self.branchA = ComplexNet()
  self.fc1 = nn.Linear(48, 3)

 def forward(self, x):
     x = self.branchA(x)
     x = self.fc1(x)
     # x = torch.relu(x)
     # x = self.fc2(x)
     # x = torch.relu(x)
     # x = self.fc3(x)
     x = torch.sigmoid(x)
     return x

CSI_train = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp3\\CSI_train.mat')['CSI_train']
CSI_train = torch.from_numpy(np.array(CSI_train, dtype=np.complex64))
y_train = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp3\\Y_train.mat')['Y_train']
y_train = torch.from_numpy(np.array(y_train-1, dtype=int))
# y_train = F.one_hot(y_train.long(),num_classes=3)
y_train = y_train.long()
y_train = torch.squeeze(y_train)

CSI_test = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp3\\CSComI_test.mat')['CSI_test']
CSI_test = torch.from_numpy(np.array(CSI_test, dtype=np.complex64))
y_test = scio.loadmat('F:\穿墙WIFI\实验对比方法\\0myMethod\\exp3\\Y_test.mat')['Y_test']
y_test = torch.from_numpy(np.array(y_test-1, dtype=np.int))
# y_test = F.one_hot(y_test.long(),num_classes=3)
y_test = y_test.long()
y_test = torch.squeeze(y_test)

train_set = WiFiDataset(CSI_train, y_train)
test_set = WiFiDataset(CSI_test, y_test)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

net = WiFiNet3().cuda(0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


for epoch in range(epoch_num):  # loop over the dataset multiple times
   train_acc = 0.0
   train_loss = 0.0
   test_acc = 0.0
   test_loss = 0.0

   net.train()
   for i, (csi, label) in enumerate(train_loader):
       optimizer.zero_grad()
       train_pred = net(csi.cuda(0))
       batch_loss = criterion(train_pred, label.cuda(0))
       batch_loss.backward()
       optimizer.step()

       train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == label.numpy())
       train_loss += batch_loss.item()

   net.eval()
   with torch.no_grad():
      for i, (csi, label) in enumerate(test_loader):
         test_pred = net(csi.cuda(0))
         batch_loss = criterion(test_pred, label.cuda(0))

         test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == label.numpy())
         test_loss += batch_loss.item()
      print('epoch %d: train loss: %.4f | test loss %.4f | train acc %.4f | test acc %.4f' %(epoch, train_loss/train_set.__len__(),test_loss/test_set.__len__(),train_acc/train_set.__len__(),test_acc/test_set.__len__()))

# 画出混淆矩阵
   net.eval()

   gt_label = np.array([])
   pred_label = np.array([])
   with torch.no_grad():
      for i, (csi,  label) in enumerate(test_loader):
         test_pred = net(csi.cuda(0))
         test_pred = torch.argmax(test_pred, dim=1)


         gt_label = np.concatenate((gt_label, label.numpy()))
         pred_label = np.concatenate((pred_label, test_pred.cpu().numpy()))
      conf_matrix = confusion_matrix(gt_label, pred_label)
      plt.figure(figsize=(8, 6))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                  xticklabels=[f'Class {i}' for i in range(3)],
                  yticklabels=[f'Class {i}' for i in range(3)])
      plt.xlabel('Predicted Labels')
      plt.ylabel('True Labels')
      plt.title('Confusion Matrix')
      plt.show()