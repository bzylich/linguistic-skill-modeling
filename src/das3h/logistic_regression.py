import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error


def spy_sparse2torch_sparse(data):
    """

    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    # print(samples, features, flush=True)
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


class LogisticRegression(nn.Module):

    def __init__(self, num_labels, num_features, use_cuda=False):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(num_features, num_labels)
        # self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
        self.use_cuda = use_cuda

    def forward(self, features):
        return self.linear(features)
        # return self.sigmoid(self.linear(features))

    def evaluate(self, features):
        self.eval()
        with torch.no_grad():
            return self.linear(features)

    def predict(self, features):
        self.eval()
        with torch.no_grad():
            features = spy_sparse2torch_sparse(features)
            if self.use_cuda:
                features = features.cuda()
            return self.sigmoid(self.linear(features))

    # def predict_softmax(self, features):
    #     self.eval()
    #     with torch.no_grad():
    #         features = spy_sparse2torch_sparse(features)
    #         if self.use_cuda:
    #             features = features.cuda()
    #         return F.softmax(self.linear(features), dim=1)


def loss_with_z_term(loss_fn, z_hat, y, class_weights=None, seen=None, z_weight=1.0, eps=1e-6):
    y_clamp = torch.clamp(y, eps, 1.0 - eps)
    z = torch.log(y_clamp / (1-y_clamp))
    if seen is not None:
        return torch.mean((loss_fn(z_hat, y).flatten() + z_weight * torch.square(z - z_hat).flatten()) * seen)
    else:
        if class_weights is not None:
            weight_indices = torch.floor(y / 0.1).long().view(-1)
            weight_indices[weight_indices == 10] = 9
            # print(weight_indices.size(), flush=True)
            class_weights_to_apply = class_weights[weight_indices]
            # print(class_weights_to_apply.size(), flush=True)
            loss_intermediate = loss_fn(z_hat, y).view(-1) + z_weight * torch.square(z - z_hat).view(-1)
            # print(loss_intermediate.size(), flush=True)
            return torch.mean(loss_intermediate * class_weights_to_apply)
        else:
            return loss_fn(z_hat, y) + z_weight * torch.mean(torch.square(z - z_hat))


def kl_ber(p, q):
    return p * (torch.log(p / q)) + (1 - p) * (torch.log((1 - p) / (1 - q)))


def kl_ber_sym(p, q, eps=1e-6):
    p_clamp = torch.clamp(p, eps, 1.0 - eps)
    q_clamp = torch.clamp(q, eps, 1.0 - eps)
    return torch.mean(kl_ber(p_clamp, q_clamp) + kl_ber(q_clamp, p_clamp)) / 2


def ce(p, q):
    return - (p * torch.log(q) + (1 - p) * torch.log(1 - q))


def ce_sym(p, q, eps=1e-6):
    p_clamp = torch.clamp(p, eps, 1.0 - eps)
    q_clamp = torch.clamp(q, eps, 1.0 - eps)
    return torch.mean(ce(p_clamp, q_clamp) + ce(q_clamp, p_clamp)) / 2


# def binomial_NLL(p, k, n):
#     return torch.mean(-k * torch.log(p) - (n - k) * torch.log(1 - p))


def create_binomial_NLL(loss_compute):
    def binomial_NLL(z_hat, y, n):
        loss_vals = loss_compute(z_hat, y).flatten()
        # print(loss_vals.size(), flush=True)
        # print(n.size(), flush=True)
        scaled = loss_vals * n
        return torch.mean(scaled)

    return binomial_NLL


def train_model(model, data, target, seen, X_dev=None, y_dev=None, seen_dev=None, class_weights=None, EPOCHS=400, LEARNING_RATE=0.1, L2_DECAY=0.0, PATIENCE=3, Z_WEIGHT=1.0, batch_size=None, cuda=False):
    dev = X_dev is not None and y_dev is not None

    if seen is None and class_weights is None:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        # loss_function = create_binomial_NLL(nn.BCEWithLogitsLoss(reduction='none'))
        loss_function = nn.BCEWithLogitsLoss(reduction='none')

    # loss_function = kl_ber_sym
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY)

    if batch_size is None:
        data = spy_sparse2torch_sparse(data)
    target = torch.FloatTensor(target)
    if seen is not None:
        seen = torch.FloatTensor(seen)

    if cuda and batch_size is None:
        data = data.cuda()
        target = target.cuda()
        if seen is not None:
            seen = seen.cuda()

    if dev:
        X_dev = spy_sparse2torch_sparse(X_dev)
        y_dev_cpu = y_dev.cpu()
        # y_dev = torch.FloatTensor(y_dev)

        if cuda:
            X_dev = X_dev.cuda()
            # y_dev = y_dev.cuda()

    min_dev_loss = None
    patience_attempts = 0
    start_time = time.time()
    for epoch in range(EPOCHS):
        if batch_size is not None:
            permutation = torch.randperm(data.shape[0])
            for i in range(0, data.shape[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i: i + batch_size]
                batch_data = data[indices]
                batch_data = spy_sparse2torch_sparse(batch_data)
                batch_target = target[indices]
                if seen is not None:
                    batch_seen = seen[indices]

                if cuda:
                    batch_data = batch_data.cuda()
                    batch_target = batch_target.cuda()
                    if seen is not None:
                        batch_seen = batch_seen.cuda()

                model.train()
                z_hat = model(batch_data)  # [:, 1]
                # loss = loss_function(z_hat, target)  # .unsqueeze(1))
                if seen is not None:
                    loss = loss_with_z_term(loss_function, z_hat, batch_target, class_weights=class_weights, seen=batch_seen, z_weight=Z_WEIGHT)
                else:
                    loss = loss_with_z_term(loss_function, z_hat, batch_target, class_weights=class_weights, z_weight=Z_WEIGHT)
                loss.backward()
                optimizer.step()

                if cuda:
                    batch_data = batch_data.cpu()
                    batch_target = batch_target.cpu()
                    if seen is not None:
                        batch_seen = batch_seen.cpu()
        else:
            optimizer.zero_grad()

            model.train()
            z_hat = model(data)  # [:, 1]
            # loss = loss_function(z_hat, target)  # .unsqueeze(1))
            loss = loss_with_z_term(loss_function, z_hat, target, class_weights=class_weights, seen=seen, z_weight=Z_WEIGHT)
            loss.backward()
            optimizer.step()

        if dev:
            dev_z_hat = model.evaluate(X_dev)  # [:, 1]
            # dev_loss = loss_function(dev_z_hat, y_dev)  # .unsqueeze(1))
            dev_loss = loss_with_z_term(loss_function, dev_z_hat, y_dev, class_weights=class_weights, seen=seen_dev, z_weight=Z_WEIGHT)

            if min_dev_loss is None or float(dev_loss) < min_dev_loss:
                min_dev_loss = float(dev_loss)
                patience_attempts = 0
            else:
                patience_attempts += 1
                if patience_attempts >= PATIENCE:
                    break

        if epoch % 5 == 0:
            if dev:
                preds = torch.sigmoid(dev_z_hat).cpu()
                dev_auc = roc_auc_score(np.round(y_dev_cpu), preds)
                dev_mae = mean_absolute_error(y_dev_cpu, preds)
                elapsed = (time.time() - start_time) / 5
                print("loss at epoch", epoch, ":", float(loss), "dev_loss at epoch", epoch, ":", float(dev_loss), ", dev_auc:", dev_auc, "dev_mae:", dev_mae, ",", elapsed, "s elapsed per epoch", flush=True)
            else:
                print("loss at epoch", epoch, ":", float(loss), ",", elapsed, "s elapsed per epoch", flush=True)
            start_time = time.time()

    print("finished at epoch", epoch, flush=True)

    if dev:
        print("final loss:", float(loss), "final dev_loss:", float(dev_loss), flush=True)
    else:
        print("final loss:", float(loss), flush=True)

    return float(dev_loss) if dev else float(loss), loss_function
