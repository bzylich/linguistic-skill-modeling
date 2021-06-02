from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error
# from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression, train_model, spy_sparse2torch_sparse, loss_with_z_term, ce_sym, kl_ber_sym
from scipy.sparse import load_npz, hstack, csr_matrix
import pandas as pd
import pywFM
import argparse
import numpy as np
import os
import sys
import dataio
import json
import torch
import matplotlib.pyplot as plt
import copy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = os.path.join(os.path.dirname(__file__),
                                        'libfm/bin/')

parser = argparse.ArgumentParser(description='Run DAS3H.')
parser.add_argument('X_file', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='assistments12')
parser.add_argument('--tags', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--lemma', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--subword_skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--vocab_size', type=int, nargs='?', default=500)
parser.add_argument('--nbest', type=int, nargs='?', default=10)
parser.add_argument('--duo_split', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--grid_search', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--continuous_correct', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--weight_by_seen', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--cuda', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--d', type=int, nargs='?')
parser.add_argument('--iter', type=int, nargs='?', default=300)

# parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--skills', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--wins', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--fails', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--attempts', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--tw_kc', type=bool, nargs='?', const=True, default=False)
# parser.add_argument('--tw_items', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

# class_weights = torch.from_numpy(np.load("./data/duolingo_hlr/class_distribution.npy"))
minority_weight = 1
class_weights = torch.ones(10, dtype=torch.float32) * minority_weight
class_weights[-1] = 1
print(class_weights, flush=True)
if options.cuda:
    class_weights = class_weights.cuda()

print("weight by seen:", options.weight_by_seen, flush=True)

experiment_args = vars(options)
DATASET_NAME = options.dataset
print("Dataset:", options.X_file, flush=True)
CSV_FOLDER = dataio.build_new_paths(DATASET_NAME)

# Build legend
short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(experiment_args)

EXPERIMENT_FOLDER = os.path.join(CSV_FOLDER, "results", short_legend)
dataio.prepare_folder(EXPERIMENT_FOLDER)
maxRuns = 5
for run_id in range(maxRuns):
    dataio.prepare_folder(os.path.join(EXPERIMENT_FOLDER, str(run_id)))

# Load sparsely encoded datasets
X = csr_matrix(load_npz(options.X_file))
all_users = np.unique(X[:, 0].toarray().flatten())
# y = X[:, 3].toarray().flatten()

modifier = ""
if options.subword_skills:
    submodifier = ""
    if options.tags:
        submodifier += "_tags"
    if options.lemma:
        submodifier += "_lemma"
    modifier = "_subword"+submodifier+"_vocab="+str(options.vocab_size)+"_nbest="+str(options.nbest)
elif options.tags:
    modifier = "_tags"
    if options.lemma:
        modifier += "_lemma"
elif options.lemma:
    modifier = "_lemma"

if options.continuous_correct:
    modifier += "_continuous"

qmat = load_npz(os.path.join(CSV_FOLDER, "q_mat"+modifier+".npz"))

print("q matrix shape:", np.shape(qmat), flush=True)

# FM parameters
params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}

if options.duo_split:
    if options.grid_search:
        splitpoint1 = int(0.8 * X.shape[0])
        splitpoint2 = int(0.9 * X.shape[0])

        X_train = X[:splitpoint1]
        y_train = X_train[:, 3].toarray()
        # successes_train = X_train[:, 5].toarray().flatten()

        X_dev = X[splitpoint1:splitpoint2]
        y_dev = X_dev[:, 3].toarray()
        # successes_dev = X_dev[:, 5].toarray().flatten()

        X_test = X[splitpoint2:]
        y_test = X_test[:, 3].toarray()
        # successes_test = X_test[:, 5].toarray().flatten()

        if options.weight_by_seen:
            seen_train = X_train[:, 6].toarray().flatten()
            seen_dev = X_dev[:, 6].toarray().flatten()
            seen_test = X_test[:, 6].toarray().flatten()
        else:
            seen_train = None
            seen_dev = None
            seen_test = None

    else:
        # splitpoint = int(0.9 * X.shape[0])
        #
        # X_train = X[:splitpoint]
        # y_train = X_train[:, 3].toarray()
        # X_dev = None
        # y_dev = None
        # X_test = X[splitpoint:]
        # y_test = X_test[:, 3].toarray()

        splitpoint1 = int(0.8 * X.shape[0])
        splitpoint2 = int(0.9 * X.shape[0])

        X_train = X[:splitpoint1]
        y_train = X_train[:, 3].toarray()
        # successes_train = X_train[:, 5].toarray().flatten()

        X_dev = X[splitpoint1:splitpoint2]
        y_dev = X_dev[:, 3].toarray()
        # successes_dev = X_dev[:, 5].toarray().flatten()

        X_test = X[splitpoint2:]
        y_test = X_test[:, 3].toarray()
        # successes_test = X_test[:, 5].toarray().flatten()

        if options.weight_by_seen:
            seen_train = X_train[:, 6].toarray().flatten()
            seen_dev = X_dev[:, 6].toarray().flatten()
            seen_test = X_test[:, 6].toarray().flatten()
        else:
            seen_train = None
            seen_dev = None
            seen_test = None

    y_train_median = np.median(y_train)
    y_train_average = np.mean(y_train)
    print("y_train median:", y_train_median, "mean:", y_train_average)
    print("y_dev median:", np.median(y_dev), "mean:", np.mean(y_dev))
    print("y_test median:", np.median(y_test), "mean:", np.mean(y_test))

    if options.d == 0:
        # y_train_lr = np.hstack((1-y_train, y_train))
        # y_test_lr = np.hstack((1-y_test, y_test))
        #
        # # if options.grid_search:
        # y_dev_lr = np.hstack((1-y_dev, y_dev))

        y_train_lr = copy.deepcopy(y_train)
        y_test_lr = copy.deepcopy(y_test)
        y_dev_lr = copy.deepcopy(y_dev)

    y_train = y_train.flatten()
    y_test = y_test.flatten()
    if options.grid_search:
        y_dev = y_dev.flatten()

    print("X_train", np.shape(X_train), flush=True)
    print("y_train", np.shape(y_train), flush=True)
    print("X_dev", np.shape(X_dev), flush=True)
    print("y_dev", np.shape(y_dev), flush=True)
    print("X_test", np.shape(X_test), flush=True)
    print("y_test", np.shape(y_test), flush=True)

    if options.d == 0:
        print('fitting...', flush=True)
        # model = LogisticRegression(solver="lbfgs", max_iter=400)
        # model.fit(X_train[:, 7:], y_train)  # the 5 first columns are the non-sparse dataset
        # y_pred_test = model.predict_proba(X_test[:, 7:])[:, 1]

        if options.grid_search:

            grid_epoch = [200000]
            grid_lr = [1e-3]
            # grid_lr = [1e-5, 1e-4]
            grid_l2 = [0.0, 0.05]
            grid_z_weight = [0.0, 0.01, 0.1]
            grid_batch_sizes = [None] if options.cuda else [None]
            # grid_z_weight = [0]
            PATIENCE = 5

            # best_loss = None
            best_dev_mae = None
            best_parameters = None

            best_dev_auc = None
            best_dev_auc_parameters = None

            best_dev_kl = None
            best_dev_kl_parameters = None

            best_dev_ce = None
            best_dev_ce_parameters = None

            X_test_loss = spy_sparse2torch_sparse(X_test[:, 7:])
            y_test_lr = torch.FloatTensor(y_test_lr)
            y_dev_lr = torch.FloatTensor(y_dev_lr)

            if options.weight_by_seen:
                seen_test = torch.FloatTensor(seen_test)
                seen_dev = torch.FloatTensor(seen_dev)

            if options.cuda:
                X_test_loss = X_test_loss.cuda()
                # y_test_lr = y_test_lr.cuda()

            for num_epochs in grid_epoch:
                for lr in grid_lr:
                    for batch_size in grid_batch_sizes:
                        for l2 in grid_l2:
                            for z_weight in grid_z_weight:

                                print("(num epochs:", num_epochs, "lr:", lr, "bs:", batch_size, "l2:", l2, "z_weight:", z_weight, ")-----------------------------------------", flush=True)
                                model = LogisticRegression(1, X_train[:, 7:].shape[1], use_cuda=options.cuda)
                                print("model parameters:", count_parameters(model), flush=True)
                                if options.cuda:
                                    model.cuda()
                                    y_dev_lr = y_dev_lr.cuda()
                                    if options.weight_by_seen:
                                        seen_dev = seen_dev.cuda()
                                dev_loss, loss_function = train_model(model, X_train[:, 7:], y_train_lr, seen_train, class_weights=class_weights, X_dev=X_dev[:, 7:], y_dev=y_dev_lr, seen_dev=seen_dev, EPOCHS=num_epochs, LEARNING_RATE=lr, L2_DECAY=l2, Z_WEIGHT=z_weight, batch_size=batch_size, PATIENCE=PATIENCE, cuda=options.cuda)
                                # if best_loss is None or dev_loss < best_loss:
                                #     best_loss = dev_loss
                                #     best_parameters = (num_epochs, lr, l2)

                                y_dev_lr = y_dev_lr.cpu()
                                y_pred_dev = model.predict(X_dev[:, 7:]).cpu() #[:, 1].cpu()
                                # y_pred_dev_softmax = model.predict_softmax(X_dev[:, 7:]).cpu() #[:, 1].cpu()
                                # print(y_pred_dev)

                                ACC = accuracy_score(np.round(y_dev), np.round(y_pred_dev))
                                print("dev acc", ACC, flush=True)
                                AUC = roc_auc_score(np.round(y_dev), y_pred_dev)
                                print('dev auc', AUC, flush=True)

                                if best_dev_auc is None or AUC > best_dev_auc:
                                    best_dev_auc = AUC
                                    best_dev_auc_parameters = (num_epochs, lr, batch_size, l2, z_weight)

                                # NLL = log_loss(y_test, y_pred_test)
                                # print('nll', NLL, flush=True)
                                MAE = mean_absolute_error(y_dev, y_pred_dev)
                                print('dev mae', MAE, flush=True)

                                if best_dev_mae is None or MAE < best_dev_mae:
                                    best_dev_mae = MAE
                                    best_parameters = (num_epochs, lr, batch_size, l2, z_weight)

                                # kl_ber = kl_ber_sym(y_dev_lr, y_pred_dev)
                                # print("dev kl_ber", float(kl_ber), flush=True)
                                #
                                # if best_dev_kl is None or kl_ber < best_dev_kl:
                                #     best_dev_kl = float(kl_ber)
                                #     best_dev_kl_parameters = (num_epochs, lr, batch_size, l2, z_weight)
                                #
                                # ce = ce_sym(y_dev_lr, y_pred_dev)
                                # print("dev ce", float(ce), flush=True)
                                #
                                # if best_dev_ce is None or ce < best_dev_ce:
                                #     best_dev_ce = float(ce)
                                #     best_dev_ce_parameters = (num_epochs, lr, batch_size, l2, z_weight)

                                # MAE = mean_absolute_error(y_dev, torch.ones_like(y_pred_dev))
                                # print('dev- all ones mae:', MAE, flush=True)

                                # ACC_sm = accuracy_score(np.round(y_dev), np.round(y_pred_dev_softmax))
                                # print("dev acc_sm", ACC_sm, flush=True)
                                # AUC_sm = roc_auc_score(np.round(y_dev), y_pred_dev_softmax)
                                # print('dev auc_sm', AUC_sm, flush=True)
                                # # NLL = log_loss(y_test, y_pred_test)
                                # # print('nll', NLL, flush=True)
                                # MAE_sm = mean_absolute_error(y_dev, y_pred_dev_softmax)
                                # print('dev mae_sm', MAE_sm, flush=True)

                                test_loss = loss_with_z_term(loss_function, model.evaluate(X_test_loss).cpu(), y_test_lr, seen=seen_test, class_weights=class_weights.cpu(), z_weight=z_weight)
                                print("test_loss", float(test_loss), flush=True)

                                y_pred_test = model.predict(X_test[:, 7:]).cpu() #[:, 1].cpu()
                                # y_pred_test_softmax = model.predict_softmax(X_test[:, 7:]).cpu() #[:, 1].cpu()
                                ACC = accuracy_score(np.round(y_test), np.round(y_pred_test))
                                print("acc", ACC, flush=True)
                                AUC = roc_auc_score(np.round(y_test), y_pred_test)
                                print('auc', AUC, flush=True)
                                # NLL = log_loss(y_test, y_pred_test)
                                # print('nll', NLL, flush=True)
                                MAE = mean_absolute_error(y_test, y_pred_test)
                                print('mae', MAE, flush=True)

                                # kl_ber = kl_ber_sym(y_test_lr, y_pred_test)
                                # print("kl_ber", float(kl_ber), flush=True)
                                #
                                # ce = ce_sym(y_test_lr, y_pred_test)
                                # print("ce", float(ce), flush=True)

                                # MAE = mean_absolute_error(y_test, torch.ones_like(y_pred_test))
                                # print('test- all ones mae:', MAE, flush=True)
                                #
                                # test_average = np.mean(y_test)
                                #
                                # MAE = mean_absolute_error(y_test, torch.ones_like(y_pred_test) * y_train_average)
                                # print('test- all train average (', y_train_average, ') mae:', MAE, flush=True)
                                #
                                # test_median = np.median(y_test)
                                #
                                # MAE = mean_absolute_error(y_test, torch.ones_like(y_pred_test) * y_train_median)
                                # print('test- all train median (', y_train_median, ') mae:', MAE, flush=True)


                                # ACC_sm = accuracy_score(np.round(y_test), np.round(y_pred_test_softmax))
                                # print("acc_sm", ACC_sm, flush=True)
                                # AUC_sm = roc_auc_score(np.round(y_test), y_pred_test_softmax)
                                # print('auc_sm', AUC_sm, flush=True)
                                # # NLL = log_loss(y_test, y_pred_test)
                                # # print('nll', NLL, flush=True)
                                # MAE_sm = mean_absolute_error(y_test, y_pred_test_softmax)
                                # print('mae_sm', MAE_sm, flush=True)


            print("best parameters (dev mae):", best_parameters, flush=True)
            print("best dev mae:", best_dev_mae, flush=True)

            print("best parameters (dev auc):", best_dev_auc_parameters, flush=True)
            print("best dev auc:", best_dev_auc, flush=True)

            # print("best parameters (dev kl):", best_dev_kl_parameters, flush=True)
            # print("best dev kl:", best_dev_kl, flush=True)
            #
            # print("best parameters (dev ce):", best_dev_ce_parameters, flush=True)
            # print("best dev ce:", best_dev_ce, flush=True)

        else:
            EPOCHS = 200000
            LR = 1e-3
            L2 = 0.05
            Z_WEIGHT = 0.1
            batch_size = None if options.cuda else None
            PATIENCE = 5

            y_dev_lr = torch.FloatTensor(y_dev_lr)
            y_test_lr = torch.FloatTensor(y_test_lr)
            if options.weight_by_seen:
                seen_test = torch.FloatTensor(seen_test)
                seen_dev = torch.FloatTensor(seen_dev)

            model = LogisticRegression(1, X_train[:, 7:].shape[1], use_cuda=options.cuda)
            print("model parameters:", count_parameters(model), flush=True)
            if options.cuda:
                model = model.cuda()
                y_dev_lr = y_dev_lr.cuda()
                if options.weight_by_seen:
                    seen_dev = seen_dev.cuda()
            dev_loss, loss_function = train_model(model, X_train[:, 7:], y_train_lr, seen_train, X_dev=X_dev[:, 7:], y_dev=y_dev_lr, seen_dev=seen_dev, class_weights=class_weights, EPOCHS=EPOCHS, LEARNING_RATE=LR, L2_DECAY=L2, Z_WEIGHT=Z_WEIGHT, batch_size=batch_size, PATIENCE=PATIENCE, cuda=options.cuda)

            X_test_loss = spy_sparse2torch_sparse(X_test[:, 7:])

            if options.cuda:
                X_test_loss = X_test_loss.cuda()
                # y_test_lr = y_test_lr.cuda()

            # test_loss = loss_function(model.evaluate(X_test_loss), y_test_lr)
            test_loss = loss_with_z_term(loss_function, model.evaluate(X_test_loss).cpu(), y_test_lr, seen=seen_test, class_weights=class_weights.cpu(), z_weight=Z_WEIGHT)
            print("test_loss", float(test_loss), flush=True)

            y_pred_test = model.predict(X_test[:, 7:]).cpu()  # [:, 1].cpu()

            # y_pred_test_softmax = model.predict_softmax(X_test[:, 7:])[:, 1].cpu()

            print(y_test, flush=True)
            print(y_pred_test, flush=True)
            ACC = accuracy_score(np.round(y_test), np.round(y_pred_test))
            print("acc", ACC, flush=True)
            AUC = roc_auc_score(np.round(y_test), y_pred_test)
            print('auc', AUC, flush=True)
            # NLL = log_loss(y_test, y_pred_test)
            # print('nll', NLL, flush=True)
            MAE = mean_absolute_error(y_test, y_pred_test)
            print('mae', MAE, flush=True)

            # kl_ber = kl_ber_sym(y_test_lr, y_pred_test)
            # print("kl_ber", float(kl_ber), flush=True)
            #
            # ce = ce_sym(y_test_lr, y_pred_test)
            # print("ce", float(ce), flush=True)

            # MAE = mean_absolute_error(y_test, torch.ones_like(y_pred_test) * y_train_average)
            # print('test- all train average (', y_train_average, ') mae:', MAE, flush=True)
            #
            # MAE = mean_absolute_error(y_test, torch.ones_like(y_pred_test) * y_train_median)
            # print('test- all train median (', y_train_median, ') mae:', MAE, flush=True)

            with open(options.X_file.replace("/duolingo_hlr/", "/duolingo_hlr/results/").replace(".npz", ".predictions.csv"), "w") as predictions_file:
                predictions_file.write("y,y_hat\n")
                for y, y_hat in zip(list(y_test), list(y_pred_test)):
                    predictions_file.write(str(y) + "," + str(float(y_hat)) + "\n")

            # ACC_sm = accuracy_score(np.round(y_test), np.round(y_pred_test_softmax))
            # print("acc_sm", ACC_sm, flush=True)
            # AUC_sm = roc_auc_score(np.round(y_test), y_pred_test_softmax)
            # print('auc_sm', AUC_sm, flush=True)
            # # NLL = log_loss(y_test, y_pred_test)
            # # print('nll', NLL, flush=True)
            # MAE_sm = mean_absolute_error(y_test, y_pred_test_softmax)
            # print('mae_sm', MAE_sm, flush=True)

            # visualize weights
            model_weights = model.linear.weight.data.cpu().numpy()
            print("weights shape:", np.shape(model_weights), flush=True)
            model_weights_print = model_weights[0]
            print("weights print shape:", np.shape(model_weights_print), flush=True)
            highest_indices = np.argsort(model_weights_print)[::-1][:1000]
            for ind in list(highest_indices):
                print(ind, model_weights_print[ind], flush=True)

            with open(options.X_file.replace("/duolingo_hlr/", "/duolingo_hlr/results/").replace(".npz", ".weights.npy"), "wb") as out_file:
                np.save(out_file, model_weights)

            # plt.matshow(model_weights)
            # plt.savefig(options.X_file.replace("/duolingo_hlr/", "/duolingo_hlr/results/").replace(".npz", ".weights.pdf"))

            # Save experimental results
            # with open(os.path.join(EXPERIMENT_FOLDER, 'duo_split', 'results.json'), 'w') as f:
            #     f.write(json.dumps({
            #         'args': experiment_args,
            #         'legends': {
            #             'short': short_legend,
            #             'full': full_legend,
            #             'latex': latex_legend
            #         },
            #         'metrics': {
            #             'ACC': ACC,
            #             'AUC': AUC,
            #             # 'NLL': NLL,
            #             'MAE': MAE
            #         }
            #     }, indent=4))

    else:
        fm = pywFM.FM(**params)
        print("X_train[:,7:]", np.shape(X_train[:, 7:]), flush=True)
        print("X_test[:,7:]", np.shape(X_test[:, 7:]), flush=True)
        model = fm.run(X_train[:, 7:], y_train, X_test[:, 7:], y_test)
        y_pred_test = np.array(model.predictions)
        model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, 'duo_split', 'rlog.csv'))

        print(y_test, flush=True)
        print(y_pred_test, flush=True)
        ACC = accuracy_score(np.round(y_test), np.round(y_pred_test))
        print("acc", ACC, flush=True)
        AUC = roc_auc_score(np.round(y_test), y_pred_test)
        print('auc', AUC, flush=True)
        # NLL = log_loss(y_test, y_pred_test)
        # print('nll', NLL, flush=True)
        MAE = mean_absolute_error(y_test, y_pred_test)
        print('mae', MAE, flush=True)


        # ACC_sm = accuracy_score(np.round(y_test), np.round(y_pred_test_softmax))
        # print("acc_sm", ACC_sm, flush=True)
        # AUC_sm = roc_auc_score(np.round(y_test), y_pred_test_softmax)
        # print('auc_sm', AUC_sm, flush=True)
        # # NLL = log_loss(y_test, y_pred_test)
        # # print('nll', NLL, flush=True)
        # MAE_sm = mean_absolute_error(y_test, y_pred_test_softmax)
        # print('mae_sm', MAE_sm, flush=True)

        # Save experimental results
        with open(os.path.join(EXPERIMENT_FOLDER, 'duo_split', 'results.json'), 'w') as f:
            f.write(json.dumps({
                'args': experiment_args,
                'legends': {
                    'short': short_legend,
                    'full': full_legend,
                    'latex': latex_legend
                },
                'metrics': {
                    'ACC': ACC,
                    'AUC': AUC,
                    # 'NLL': NLL,
                    'MAE': MAE
                }
            }, indent=4))


else:
    # Student-level train-test split
    kf = KFold(n_splits=5, shuffle=True)
    splits = kf.split(all_users)

    for run_id, (i_user_train, i_user_test) in enumerate(splits):
        users_train = all_users[i_user_train]
        users_test = all_users[i_user_test]

        X_train = X[np.where(np.isin(X[:, 0].toarray().flatten(), users_train))]
        y_train = X_train[:, 3].toarray()
        X_test = X[np.where(np.isin(X[:, 0].toarray().flatten(), users_test))]
        y_test = X_test[:, 3].toarray()

        if options.d == 0:
            y_train_lr = np.hstack((1-y_train, y_train))
            y_test_lr = np.hstack((1-y_test, y_test))

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        print("X_train", np.shape(X_train), flush=True)
        print("X_test", np.shape(X_test), flush=True)
        print("y_train", np.shape(y_train), flush=True)
        print("y_test", np.shape(y_test), flush=True)

        if options.d == 0:
            print('fitting...', flush=True)
            # model = LogisticRegression(solver="lbfgs", max_iter=400)
            # model.fit(X_train[:, 7:], y_train)  # the 5 first columns are the non-sparse dataset
            # y_pred_test = model.predict_proba(X_test[:, 7:])[:, 1]

            model = LogisticRegression(2, X_train[:, 7:].shape[1], use_cuda=options.cuda)
            train_model(model, X_train[:, 7:], y_train_lr, EPOCHS=100, LEARNING_RATE=0.1, cuda=options.cuda)
            y_pred_test = model.predict(X_test[:, 7:])[:, 1]

        else:
            fm = pywFM.FM(**params)
            print("X_train[:,7:]", np.shape(X_train[:, 7:]), flush=True)
            print("X_test[:,7:]", np.shape(X_test[:, 7:]), flush=True)
            model = fm.run(X_train[:, 7:], y_train, X_test[:, 7:], y_test)
            y_pred_test = np.array(model.predictions)
            model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'rlog.csv'))

        print(y_test, flush=True)
        print(y_pred_test, flush=True)
        ACC = accuracy_score(np.round(y_test), np.round(y_pred_test))
        AUC = roc_auc_score(np.round(y_test), y_pred_test)
        print('auc', AUC, flush=True)
        # NLL = log_loss(y_test, y_pred_test)
        # print('nll', NLL, flush=True)
        MAE = mean_absolute_error(y_test, y_pred_test)
        print('mae', MAE, flush=True)

        # Save experimental results
        with open(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'results.json'), 'w') as f:
            f.write(json.dumps({
                'args': experiment_args,
                'legends': {
                    'short': short_legend,
                    'full': full_legend,
                    'latex': latex_legend
                },
                'metrics': {
                    'ACC': ACC,
                    'AUC': AUC,
                    # 'NLL': NLL,
                    'MAE': MAE
                }
            }, indent=4))
