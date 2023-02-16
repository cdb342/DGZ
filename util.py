import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import torch.nn.functional as F
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        mx =5
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if opt.preprocessing:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.train_feature = torch.from_numpy(_train_feature).float()
            self.train_feature=F.normalize(self.train_feature,dim=-1)*mx
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature=F.normalize(self.test_unseen_feature,dim=-1)*mx
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
            self.test_seen_feature=F.normalize(self.test_seen_feature,dim=-1)*mx
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        return batch_feature, batch_label