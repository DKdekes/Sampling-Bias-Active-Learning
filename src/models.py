import pickle
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import utils
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Bert(torch.nn.Module):
    def __init__(self, opts):
        super(Bert, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(opts.pretrained_model_name,
                                                                   num_labels=opts.num_classes)
        self.opts = opts
        self.device = 'cuda'
        self.to(self.device)
        self.optimizer = AdamW(self.parameters(), lr=2e-5)
        self.softmax = torch.nn.Softmax()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True)
        return output.loss, output.logits

    def fit_(self, dset):
        dset.generate_data(train=True)
        model = self.train()
        for _ in range(self.opts.num_epochs):
            correct_predictions = 0
            n_examples = 0
            for d in tqdm(dset.train_data_loader):
                n_examples += self.opts.batch_size
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["label"].to(self.device)
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # probs = torch.nn.functional.softmax(logits, dim=0) don't need softmax for preds
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('train acc: ' + str(correct_predictions.double() / n_examples))

    def predict(self, dataloader):
        model = self.eval()
        all_probs = []
        with torch.no_grad():
            for d in tqdm(dataloader):
                input_ids = d['input_ids'].to(self.device)
                attention_mask = d['attention_mask'].to(self.device)
                _, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                probs = torch.nn.functional.softmax(logits, dim=0)
                for prob in probs:
                    all_probs.append(prob)
        all_probs = torch.stack(all_probs)
        all_probs = all_probs.cpu()
        all_probs = all_probs.numpy()
        # return the softmax probability of the predicted class for each prediction
        # todo: get better understanding of uncertainty calculation. I don't understand why we need to do the following line.
        # ret = np.array([subarray[np.argsort(index)] for subarray, index in zip(all_probs, all_preds)])
        return all_probs

    def predict_proba_(self, dset, train=True):
        """
        This is the method called in trainer to get the predicted probabilities for a set of examples.
        There are two time's this is called right now; to get the probabilities for the entire train set in order
        to perform the uncertainty query, and to get the probabilities on the test set to calculate the accuracy.
        """
        if train:
            # this dataloader will contain all the training data
            dset.generate_data(use_mask=False)
            dataloader = dset.train_data_loader
        else:
            # this dataloader will contain all the test data
            dataloader = dset.test_data_loader
        return self.predict(dataloader)

    def save_model_(self, path, itr, quantized=True):
        pass

    def get_features_(self, X):
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
