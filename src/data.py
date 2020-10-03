import numpy as np
from sklearn import preprocessing
import utils
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class Data:
    def __init__(self, opt):
        self.opt = opt
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None

    def generate_data(self, train=True):
        pass

    def get_X_y(self, train=True):
        pass


class FastTextData(Data):
    def __init__(self, opt):
        super(FastTextData, self).__init__(opt)
        self.opt = opt
        self.X, self.y = utils.get_data(opt.data_dir + opt.dataset, mode='train')
        self.X_test, self.y_test = utils.get_data(opt.data_dir + opt.dataset, mode='test')
        self.y, self.y_test = np.array(self.y), np.array(self.y_test)
        self.label_enc = preprocessing.LabelEncoder()
        self.y = self.label_enc.fit_transform(self.y)
        self.y_test = self.label_enc.transform(self.y_test)

        self.ohe_enc = preprocessing.LabelBinarizer()
        self.ohe_enc.fit(self.y)
        opt.num_classes = self.ohe_enc.classes_.shape[0]
        opt.num_points = self.y.shape[0]
        opt.num_test = self.y_test.shape[0]
        equal_labels = np.ones(opt.num_classes) * (1.0 / opt.num_classes)
        opt.best_label_entropy = -np.sum(equal_labels * np.log(equal_labels + np.finfo(float).eps))
        num_init = int(opt.init_train_percent * opt.num_points)
        self.is_train = utils.get_mask(opt.num_points, num_init)

    def generate_data(self, train=True):
        mask = self.is_train if train else ~self.is_train
        if train:
            file_path = utils.to_fastText(X=self.X, y=self.y, data_dir=self.opt.logpath, expname=self.opt.exp_name,
                                          mask=mask, mode='train')
        else:
            file_path = utils.to_fastText(X=self.X, y=self.y, data_dir=self.opt.logpath, expname=self.opt.exp_name,
                                          mask=mask, mode='pool')
        return file_path


class NaiveBayesData(Data):
    def __init__(self, opt):
        super(NaiveBayesData, self).__init__(opt)
        self.X, self.y = utils.get_data(opt.data_dir + opt.dataset, mode='train')
        self.X_test, self.y_test = utils.get_data(opt.data_dir + opt.dataset, mode='test')
        self.y, self.y_test = np.array(self.y), np.array(self.y_test)
        self.label_enc = preprocessing.LabelEncoder()
        self.y = self.label_enc.fit_transform(self.y)
        self.y_test = self.label_enc.transform(self.y_test)

        self.ohe_enc = preprocessing.LabelBinarizer()
        self.ohe_enc.fit(self.y)
        opt.num_classes = self.ohe_enc.classes_.shape[0]
        opt.num_points = self.y.shape[0]
        opt.num_test = self.y_test.shape[0]
        equal_labels = np.ones(opt.num_classes)*(1.0/opt.num_classes)
        opt.best_label_entropy = -np.sum(equal_labels * np.log(equal_labels+ np.finfo(float).eps))
        num_init = int(opt.init_train_percent * opt.num_points)
        self.is_train = utils.get_mask(opt.num_points, num_init)
    
    def get_X_y(self, train=True):
        mask = self.is_train if train else ~self.is_train
        X = [self.X[i] for i in range(mask.shape[0]) if mask[i]]
        y = self.y[mask]
        return X, y


class BertData:
    def __init__(self, opts):
        self.opts = opts
        self.X, self.y = utils.get_data(opts.data_dir + opts.dataset, mode='train')
        self.X_test, self.y_test = utils.get_data(opts.data_dir + opts.dataset, mode='test')
        self.y, self.y_test = np.array(self.y), np.array(self.y_test)
        self.label_enc = preprocessing.LabelEncoder()
        self.y = self.label_enc.fit_transform(self.y)
        self.y_test = self.label_enc.transform(self.y_test)
        self.ohe_enc = preprocessing.LabelBinarizer()
        self.ohe_enc.fit(self.y)
        opts.num_classes = self.ohe_enc.classes_.shape[0]
        opts.num_points = self.y.shape[0]
        opts.num_test = self.y_test.shape[0]
        equal_labels = np.ones(opts.num_classes) * (1.0 / opts.num_classes)
        opts.best_label_entropy = -np.sum(equal_labels * np.log(equal_labels + np.finfo(float).eps))
        num_init = int(opts.init_train_percent * opts.num_points)
        self.is_train = utils.get_mask(opts.num_points, num_init)
        self.train_data_loader = None
        self.test_data_loader = self.create_data_loader(self.X_test, self.y_test)

    def generate_data(self, train=True, use_mask=True):
        mask = self.is_train if train else ~self.is_train
        if use_mask:
            masked_X = [X for X, s in zip(self.X, mask) if s]
            self.train_data_loader = self.create_data_loader(masked_X, self.y[mask])
        else:
            self.train_data_loader = self.create_data_loader(self.X, self.y)

    def create_data_loader(self, X, y):
        ds = TransformerDataset(
            features=X,
            labels=y,
            opts=self.opts
        )
        return DataLoader(
            ds,
            batch_size=self.opts.batch_size,
            num_workers=8
        )


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, opts):
        self.features = features
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(opts.pretrained_model_name)
        self.opts = opts

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = str(self.features[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            feature,
            add_special_tokens=True,
            max_length=self.opts.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': feature,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

