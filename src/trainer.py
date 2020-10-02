import gc
import os
import numpy as np
import models


def train(logger, opt, dset, epoch, data_logger):
    logger.info(f"==> Query Iteration:{(epoch+1)}/{opt.query_iter}\tTraining Size:{dset.is_train.sum()}\tPool Size:{(~(dset.is_train)).sum()}")

    y_prob_all = np.zeros((len(dset.X), opt.num_classes))*1.0
    y_test = np.zeros((len(dset.X_test), opt.num_classes))*1.0
    y_feat = None

    for i in range(opt.num_ensemble):
        logger.debug(f"==> Loading model..")
        model = getattr(models, opt.model)(opt)
        logger.debug(f"==> Training model..")
        model.fit_(dset)
        logger.debug(f"==> Predicting..")
        y_prob_all += model.predict_proba_(dset, train=True)
        y_test += model.predict_proba_(dset, train=False)
        y_feat = model.get_features_(dset.X)

        if not os.path.exists(opt.logpath + opt.exp_name + '/pretrained/'): os.makedirs(opt.logpath + opt.exp_name + '/pretrained/')
        model.save_model_(itr=epoch, path=opt.logpath + opt.exp_name + '/pretrained/ensid_'+str(i)+'_', quantized=opt.quantize)
        del model
        gc.collect()

    data_logger.add(epoch, y_prob_all, dset.y, dset.is_train)
    y_prob_all /= (1.0*opt.num_ensemble)
    y_test /= (1.0*opt.num_ensemble)
    return y_prob_all, y_test, y_feat


def acquise(y_prob_all, y_true_all, acq, idx, itr, logger):
    y_prob, y_true = y_prob_all[idx], y_true_all[idx]
    idx, unc = acq.acquise(y_prob=y_prob)
    return idx, unc


def acquise_coreset(acq, y_feat, num_acq, train_idx, pool_idx):
    idx_batch = acq.acquise(y_feat=y_feat, num_acq=num_acq, train_idx=train_idx, pool_idx=pool_idx)
    return idx_batch