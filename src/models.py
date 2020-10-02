import pickle
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import utils
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


class Bert(torch.nn.Module):
    def __init__(self, opt):
        super(Bert, self).__init__()
        self.lm = BertModel.from_pretrained(opt.pretrained_model_name)
        self.drop = torch.nn.Dropout(p=opt.dropout)
        self.out = torch.nn.Linear(self.lm.config.hidden_size, opt.num_classes)
        self.opt = opt
        self.device = 'cuda'
        self.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        self.softmax = torch.nn.Softmax()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

    def fit_(self, dset):
        dset.generate_data(train=True)
        for _ in range(self.opt.num_epochs):
            model = self.train()
            # need to find a way to sample from the dataloader using the train mask
            for d in dset.train_data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["label"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

    def get_features_(self, X):
        # don't need to implement for now
        pass

    def predict_proba_(self, dset, train=True):
        model = self.eval()
        all_probs = []
        all_preds = []
        if train:
            # this dataloader will contain all the training data
            dset.generate_data(use_mask=False)
            dataloader = dset.train_data_loader
        else:
            # this dataloader will contain all the test data
            dataloader = dset.test_data_loader
        with torch.no_grad():
            for d in dataloader:
                input_ids = d['input_ids'].to(self.device)
                attention_mask = d['attention_mask'].to(self.device)
                labels = d['label'].to(self.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                y_proba = self.softmax(outputs)
                for pred, prob in zip(preds, y_proba):
                    all_preds.append(pred.cpu().numpy()[0])
                    all_probs.append(prob.cpu().numpy())
        # return the softmax probability of the predicted class for each prediction
        ret = np.array([subarray[np.argsort(index)] for subarray, index in zip(np.array(all_probs), np.array(all_preds))])
        return ret

    def save_model_(self, path, itr, quantized=True):
        pass


class FastText(fasttext.FastText._FastText):
    def __init__(self, opt):
        super(FastText, self).__init__()
        self.opt = opt
        self.model = None
        self.train_path = None

    def fit_(self, dset):
        self.train_path = dset.generate_data(train=True)
        self.model = fasttext.train_supervised(dim=self.opt.dim, input=self.train_path, epoch=self.opt.num_epochs,
                                            lr=self.opt.lr, wordNgrams=self.opt.num_ngrams, verbose=0,
                                            minCount=self.opt.min_count, bucket=self.opt.num_buckets,
                                               thread=self.opt.workers)
        if self.opt.quantize:
            self.quantize_(self.train_path)

    def quantize_(self, train_path):
        self.model.quantize(input=train_path, qnorm=self.opt.qnorm, retrain=self.opt.retrain_quantize,
                            cutoff=self.opt.cutoff, qout=self.opt.qout, thread=self.opt.workers)

    def get_features_(self, X):
        return np.array(list(map(self.model.get_sentence_vector, X)))

    def predict_proba_(self, dataset, train=True):
        if train:
            y_label, y_proba = self.model.predict(text=dataset.X, k=self.opt.num_classes)
        else:
            y_label, y_proba = self.model.predict(text=dataset.X_test, k=self.opt.num_classes)

        return utils.rearrange_label_proba(y_proba, y_label)

    def save_model_(self, path, itr, quantized=True):
        if quantized:
            self.model.save_model(path + f'fasttext_{itr}.ftz')
        else:
            self.model.save_model(path + f'fasttext_{itr}.bin')

    def load_model_(self, path):
        self.model = fasttext.load_model(path)


class NaiveBayes:
    def __init__(self, opt):
        self.opt = opt
        self.model = MultinomialNB()
        self.vectorizer = None

    def fit_(self, dset):
        X, y = dset.get_X_y(train=True)
        vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, stop_words='english', sublinear_tf=True).fit(X)
        X = vectorizer.transform(X)
        self.model.fit(X, y)
        self.vectorizer = vectorizer

    def get_features_(self, X):
        return self.vectorizer.transform(X)

    def predict_proba_(self, X):
        return self.model.predict_proba(self.vectorizer.transform(X))

    def save_model_(self, path, itr, quantized=True):
        with open(path + f'naivebayes_{itr}.bin', 'wb') as file:
            pickle.dump(self.model, file)
