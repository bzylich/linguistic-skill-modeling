import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]


def spy_sparse2torch_sparse(data, tensor_type=torch.sparse.FloatTensor):
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
    t = tensor_type(indices, torch.from_numpy(values).float(), [samples, features])
    return t


class LogisticRegression(nn.Module):

    def __init__(self, num_labels, num_features, feature_types, num_skills, emb_size, emb_h1_size, emb_h2_size, num_windows, dropout=0.0, embeddings=None, freeze=True, use_cuda=False):
        super(LogisticRegression, self).__init__()

        self.num_windows = num_windows
        assert num_windows == 1  # not configured yet for more than one window
        self.emb_size = emb_size
        if embeddings is None:
            self.embeddings = nn.Embedding(num_skills, emb_size, padding_idx=0, sparse=True)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=freeze, sparse=True)

        self.num_skills = num_skills
        self.feature_types = feature_types
        self.num_features = num_features

        self.emb_h1 = nn.Linear((2 * emb_size), emb_h1_size)
        self.drop1 = torch.nn.Dropout(dropout)
        self.emb_h2 = nn.Linear(emb_h1_size, emb_h2_size)
        self.drop2 = torch.nn.Dropout(dropout)
        self.emb_sim = nn.Linear(emb_h2_size, 1)
        self.emb_sim_activation = nn.Tanh()

        self.linear = nn.Linear(num_features + num_windows * self.num_skills * 2, num_labels)
        # self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
        self.use_cuda = use_cuda

    def compute_correct_total(self, skills, targets):
        num_examples, history_len = skills.size()
        # print(num_examples, history_len, flush=True)

        # skills_matched = torch.repeat_interleave(skills, history_len, dim=0)

        # concat_features = torch.zeros((num_examples * history_len, (self.emb_size * 2)), dtype=torch.float32, device=device)

        # concat_features[:, -1] = torch.arange(history_len, dtype=torch.float32, device=device).repeat(num_examples)

        # print(skills_matched.size(), hist_skills.size(), flush=True)

        # concat_features[:, :self.emb_size] = self.embeddings(torch.repeat_interleave(skills, history_len, dim=0)).float()
        # # hist_skills = torch.squeeze(hist_skills.to_dense().type(torch.LongTensor))
        # concat_features[:, self.emb_size: self.emb_size * 2] = self.embeddings(hist_skills).float()

        skill_embeddings = self.embeddings(skills).float()
        history_indices = torch.arange(history_len, device=device).repeat(history_len, 1)
        history_mask = (1 - torch.triu(torch.ones(history_len, history_len, device=device))) == 1
        history_indices_selected = history_indices[history_mask]
        adjusted_history_len = history_indices_selected.size()
        hist_skill_embeddings = skill_embeddings[:, history_indices_selected, :].view(-1, self.emb_size)

        skill_indices = torch.ones(history_len, history_len, dtype=torch.long, device=device) * torch.arange(history_len, dtype=torch.long, device=device).view(-1, 1)
        skill_embeddings = skill_embeddings[:, skill_indices[history_mask], :].view(-1, self.emb_size)

        # print(skills_emb.size(), hist_skills_emb.size(), flush=True)

        # print(skills_emb.dtype, hist_skills_emb.dtype, practice_number.dtype, flush=True)
        concat_emb = torch.cat((skill_embeddings, hist_skill_embeddings), dim=1)
        # print(concat_emb.dtype, flush=True)
        # concat_emb = torch.ones(num_examples * history_len, 600).type(torch.FloatTensor)

        hist_similarities = self.emb_sim_activation(self.emb_sim(self.drop2(F.relu(self.emb_h2(self.drop1(F.relu(self.emb_h1(concat_emb)))))))).view(num_examples, adjusted_history_len)

        total_correct_features = torch.zeros(num_examples, 2 * self.num_skills * self.num_windows, dtype=torch.float32, device=device)

        flat_skills = skills.view(-1)
        start_indices = torch.sum(history_indices * history_mask.long(), device=device, dim=1)[:-1]
        end_indices = torch.sum(history_indices * history_mask.long(), device=device, dim=1)[1:]
        total_correct_features[:, flat_skills] = torch.sum(hist_similarities[:, start_indices:end_indices], dim=1)

        # total_correct_features[:, skills] = torch.sum(hist_similarities, dim=1)

        # total_intermediate = torch.sum(hist_similarities, dim=1)
        # total_intermediate = torch.log1p(total_intermediate)

        hist_similarities *= targets[:, history_indices]
        total_correct_features[:, flat_skills+self.num_skills] = torch.sum(hist_similarities[:, start_indices:end_indices], dim=1)
        # total_correct_features[:, skills+self.num_skills] = torch.sum(hist_correct * hist_similarities, dim=1)

        # hist_correct = hist_correct.to_dense().type(torch.FloatTensor)
        # correct_intermediate = torch.sum(hist_correct * hist_similarities, dim=1)
        # correct_intermediate = torch.sum(correct_intermediate, dim=1)
        # correct_intermediate = torch.log1p(correct_intermediate)

        # row_indices = torch.arange(num_examples).type(torch.LongTensor)
        # indices = torch.cat((torch.unsqueeze(row_indices, 0), torch.unsqueeze(skills, 0)), dim=0)

        # total_features = torch.sparse.FloatTensor(indices, total_intermediate, torch.Size([num_examples, self.num_skills * self.num_windows]))
        # total_features = torch.zeros(num_examples, self.num_skills * self.num_windows, dtype=torch.float32, device=device)
        # total_features[:, skills] = total_intermediate  # num_examples x num_windows
        #
        # # correct_features = torch.sparse.FloatTensor(indices, correct_intermediate, torch.Size([num_examples, self.num_skills * self.num_windows]))
        # correct_features = torch.zeros(num_examples, self.num_skills * self.num_windows, dtype=torch.float32, device=device)
        # correct_features[:, skills] = correct_intermediate  # num_examples x num_windows
        # [:, skills:skills + self.num_windows]

        return total_correct_features
        # return torch.cat((total_intermediate, correct_intermediate), dim=0)

    def forward(self, users, items, langs, skills, targets, mask, z_weight=0.0):
        # num_examples = skills.size()[0]
        num_examples, seqlen = skills.size()
        # concat_features = torch.zeros((num_examples, num_features + (2 * self.num_skills)), dtype=torch.float32, device=device)

        total_correct_intermediate = self.compute_correct_total(skills, targets)

        # indices = torch.zeros(2, (num_examples * (self.feature_types + 2)), dtype=torch.long, device=device)
        # indices[0, :] = torch.arange(num_examples, dtype=torch.long, device=device).repeat(self.feature_types + 2)
        # indices[1, :num_examples * self.feature_types] = features  # user, item, skills, lang feature columns
        # indices[1, num_examples * self.feature_types:num_examples * (self.feature_types + 1)] = skills + self.num_features  # total feature columns
        # indices[1, num_examples * (self.feature_types + 1):] = skills + self.num_features + self.num_skills  # correct feature columns

        concat_features = torch.cat((torch.repeat_interleave(users, seqlen, dim=0), items, langs, to_categorical(skills.view(-1), self.num_skills), total_correct_intermediate), dim=1)  # num_examples x (num_features + num_count_features)
        # values = torch.ones(num_examples * (self.feature_types + 2), dtype=torch.float32, device=device)
        # values[num_examples * self.feature_types:] = total_correct_intermediate

        # concat_features = torch.sparse.FloatTensor(indices, values, torch.Size([num_examples, self.num_features + self.num_windows * self.num_skills * 2]))

        # print(concat_features.size(), flush=True)

        logits = self.linear(concat_features)
        loss_function = nn.BCEWithLogitsLoss()

        mask = mask.view(-1)
        preds = logits.view(-1)[mask]
        labels = targets.view(-1)[mask]

        loss = loss_with_z_term(loss_function, preds, labels, z_weight=z_weight)
        return loss, self.sigmoid(logits)
        # return self.sigmoid(self.linear(features))

    def evaluate(self, features, skills, hist_skills, hist_correct):
        self.eval()
        with torch.no_grad():
            correct_features, total_features = self.compute_correct_total(skills, hist_skills, hist_correct)

            concat_features = torch.cat((features, correct_features, total_features), dim=1)  # num_examples x (num_features + num_count_features)

            return self.linear(concat_features)

    def predict(self, features, skills, hist_skills, hist_correct):
        self.eval()
        with torch.no_grad():
            correct_features, total_features = self.compute_correct_total(skills, hist_skills, hist_correct)

            concat_features = torch.cat((features, correct_features, total_features), dim=1)  # num_examples x (num_features + num_count_features)

            return self.sigmoid(self.linear(concat_features))

    # def predict_softmax(self, features):
    #     self.eval()
    #     with torch.no_grad():
    #         features = spy_sparse2torch_sparse(features)
    #         if self.use_cuda:
    #             features = features.cuda()
    #         return F.softmax(self.linear(features), dim=1)


def loss_with_z_term(loss_fn, z_hat, y, seen=None, z_weight=1.0, eps=1e-6):
    y_clamp = torch.clamp(y, eps, 1.0 - eps)
    z = torch.log(y_clamp / (1-y_clamp))

    y_view = y.view(-1, 1)
    if seen is not None:
        return torch.mean((loss_fn(z_hat, y_view).flatten() + z_weight * torch.square(z - z_hat).flatten()) * seen)
    else:
        return loss_fn(z_hat, y_view) + z_weight * torch.mean(torch.square(z - z_hat))


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


def train_model(model, data, target, seen, skills, hist_skills, hist_correct, X_dev=None, y_dev=None, seen_dev=None, skills_dev=None, hist_skills_dev=None, hist_correct_dev=None, EPOCHS=400, LEARNING_RATE=0.1, L2_DECAY=0.0, PATIENCE=3, Z_WEIGHT=1.0, batch_size=None, cuda=False):
    dev = X_dev is not None and y_dev is not None

    if seen is None:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        # loss_function = create_binomial_NLL(nn.BCEWithLogitsLoss(reduction='none'))
        loss_function = nn.BCEWithLogitsLoss(reduction='none')

    # loss_function = kl_ber_sym
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY)

    if batch_size is None:
        data = spy_sparse2torch_sparse(data)
        hist_skills = spy_sparse2torch_sparse(hist_skills.reshape(-1, 1), tensor_type=torch.sparse.LongTensor)
        hist_correct = spy_sparse2torch_sparse(hist_correct)
    target = torch.FloatTensor(target)
    if seen is not None:
        seen = torch.FloatTensor(seen)

    if cuda and batch_size is None:
        data = data.cuda()
        target = target.cuda()
        if seen is not None:
            seen = seen.cuda()

    if dev:
        # X_dev = spy_sparse2torch_sparse(X_dev)
        # y_dev = torch.FloatTensor(y_dev)

        if cuda:
            X_dev = X_dev.cuda()
            # y_dev = y_dev.cuda()

    print("about to start training...", flush=True)
    min_dev_loss = None
    patience_attempts = 0
    start_time = time.time()
    batch_start_time = time.time()
    for epoch in range(EPOCHS):
        if batch_size is not None:
            permutation = torch.randperm(data.shape[0])
            batch_count = 0
            for i in range(0, data.shape[0], batch_size):
                batch_count += 1
                optimizer.zero_grad()

                indices = permutation[i: i + batch_size]

                batch_data = data[indices]
                # batch_data = spy_sparse2torch_sparse(batch_data)
                batch_data = torch.FloatTensor(batch_data.toarray())

                batch_target = target[indices]
                batch_skills = skills[indices]

                batch_hist_skills = hist_skills[indices]
                batch_hist_correct = hist_correct[indices]
                # remove extra padding
                columns_to_keep = batch_hist_skills.getnnz(0) > 0
                batch_hist_correct = batch_hist_correct[:, columns_to_keep]
                batch_hist_skills = batch_hist_skills[:, columns_to_keep]
                # batch_hist_skills = spy_sparse2torch_sparse(batch_hist_skills.reshape(-1, 1), tensor_type=torch.sparse.LongTensor)
                batch_hist_skills = torch.LongTensor(batch_hist_skills.toarray().flatten())
                # batch_hist_correct = spy_sparse2torch_sparse(batch_hist_correct)
                batch_hist_correct = torch.FloatTensor(batch_hist_correct.toarray())

                if seen is not None:
                    batch_seen = seen[indices]

                if cuda:
                    batch_data = batch_data.cuda()
                    batch_target = batch_target.cuda()
                    batch_skills = batch_skills.cuda()
                    batch_hist_skills = batch_hist_skills.cuda()
                    batch_hist_correct = batch_hist_correct.cuda()
                    if seen is not None:
                        batch_seen = batch_seen.cuda()

                model.train()
                z_hat = model(batch_data, batch_skills, batch_hist_skills, batch_hist_correct)  # [:, 1]
                # loss = loss_function(z_hat, target)  # .unsqueeze(1))
                if seen is not None:
                    loss = loss_with_z_term(loss_function, z_hat, batch_target, seen=batch_seen, z_weight=Z_WEIGHT)
                else:
                    loss = loss_with_z_term(loss_function, z_hat, batch_target, z_weight=Z_WEIGHT)
                loss.backward()
                optimizer.step()

                if cuda:
                    batch_data = batch_data.cpu()
                    batch_target = batch_target.cpu()
                    batch_skills = batch_skills.cpu()
                    batch_hist_skills = batch_hist_skills.cpu()
                    batch_hist_correct = batch_hist_correct.cpu()
                    if seen is not None:
                        batch_seen = batch_seen.cpu()

                if batch_count % 5 == 0:
                    batch_elapsed = (time.time() - batch_start_time) / 5
                    print("loss at batch", batch_count, ":", float(loss), ",", batch_elapsed, "s elapsed per batch", flush=True)
                    batch_start_time = time.time()
                if batch_count % 500 == 499:
                    if dev:
                        dev_z_hats = None
                        for j in range(0, X_dev.shape[0], batch_size):
                            batch_X_dev = torch.FloatTensor(X_dev[j: j + batch_size].toarray())
                            batch_skills_dev = skills_dev[j: j + batch_size]
                            batch_hist_skills_dev = hist_skills_dev[j: j + batch_size]
                            batch_hist_correct_dev = hist_correct_dev[j: j + batch_size]
                            columns_to_keep = batch_hist_skills_dev.getnnz(0) > 0
                            batch_hist_skills_dev = torch.LongTensor(batch_hist_skills_dev[:, columns_to_keep].toarray().flatten())
                            batch_hist_correct_dev = torch.FloatTensor(batch_hist_correct_dev[:, columns_to_keep].toarray())

                            if cuda:
                                batch_X_dev = batch_X_dev.cuda()
                                batch_skills_dev = batch_skills_dev.cuda()
                                batch_hist_skills_dev = batch_hist_skills_dev.cuda()
                                batch_hist_correct_dev = batch_hist_correct_dev.cuda()

                            dev_z_hat = model.evaluate(batch_X_dev, batch_skills_dev, batch_hist_skills_dev, batch_hist_correct_dev)  # [:, 1]
                            if dev_z_hats is None:
                                dev_z_hats = dev_z_hat
                            else:
                                dev_z_hats = torch.cat((dev_z_hats, dev_z_hat), dim=0)

                            if cuda:
                                batch_X_dev = batch_X_dev.cpu()
                                batch_skills_dev = batch_skills_dev.cpu()
                                batch_hist_skills_dev = batch_hist_skills_dev.cpu()
                                batch_hist_correct_dev = batch_hist_correct_dev.cpu()

                        # dev_loss = loss_function(dev_z_hat, y_dev)  # .unsqueeze(1))
                        dev_z_hats = dev_z_hats.cpu()
                        dev_loss = loss_with_z_term(loss_function, dev_z_hats, y_dev, seen=seen_dev, z_weight=Z_WEIGHT)

                        if min_dev_loss is None or float(dev_loss) < min_dev_loss:
                            min_dev_loss = float(dev_loss)
                            patience_attempts = 0
                        else:
                            patience_attempts += 1
                            if patience_attempts >= PATIENCE:
                                break

                        print("loss at epoch", epoch, "batch", batch_count, ":", float(loss), "dev_loss at epoch", epoch, ":", float(dev_loss), flush=True)

        else:
            optimizer.zero_grad()

            model.train()
            z_hat = model(data, skills, hist_skills, hist_correct)  # [:, 1]
            # loss = loss_function(z_hat, target)  # .unsqueeze(1))
            loss = loss_with_z_term(loss_function, z_hat, target, seen=seen, z_weight=Z_WEIGHT)
            loss.backward()
            optimizer.step()

        if dev:
            if patience_attempts > PATIENCE:
                break
            dev_z_hats = None
            for j in range(0, X_dev.shape[0], batch_size):
                batch_X_dev = torch.FloatTensor(X_dev[j: j + batch_size].toarray())
                batch_skills_dev = skills_dev[j: j + batch_size]
                batch_hist_skills_dev = hist_skills_dev[j: j + batch_size]
                batch_hist_correct_dev = hist_correct_dev[j: j + batch_size]
                columns_to_keep = batch_hist_skills_dev.getnnz(0) > 0
                batch_hist_skills_dev = torch.LongTensor(batch_hist_skills_dev[:, columns_to_keep].toarray().flatten())
                batch_hist_correct_dev = torch.FloatTensor(batch_hist_correct_dev[:, columns_to_keep].toarray())

                if cuda:
                    batch_X_dev = batch_X_dev.cuda()
                    batch_skills_dev = batch_skills_dev.cuda()
                    batch_hist_skills_dev = batch_hist_skills_dev.cuda()
                    batch_hist_correct_dev = batch_hist_correct_dev.cuda()

                dev_z_hat = model.evaluate(batch_X_dev, batch_skills_dev, batch_hist_skills_dev, batch_hist_correct_dev)  # [:, 1]
                if dev_z_hats is None:
                    dev_z_hats = dev_z_hat
                else:
                    dev_z_hats = torch.cat((dev_z_hats, dev_z_hat), dim=0)

                if cuda:
                    batch_X_dev = batch_X_dev.cpu()
                    batch_skills_dev = batch_skills_dev.cpu()
                    batch_hist_skills_dev = batch_hist_skills_dev.cpu()
                    batch_hist_correct_dev = batch_hist_correct_dev.cpu()

            # dev_loss = loss_function(dev_z_hat, y_dev)  # .unsqueeze(1))
            dev_z_hats = dev_z_hats.cpu()
            dev_loss = loss_with_z_term(loss_function, dev_z_hats, y_dev, seen=seen_dev, z_weight=Z_WEIGHT)

            if min_dev_loss is None or float(dev_loss) < min_dev_loss:
                min_dev_loss = float(dev_loss)
                patience_attempts = 0
            else:
                patience_attempts += 1
                if patience_attempts >= PATIENCE:
                    break

        # if epoch % 5 == 0:
        elapsed = (time.time() - start_time) / 5
        if dev:
            print("loss at epoch", epoch, ":", float(loss), "dev_loss at epoch", epoch, ":", float(dev_loss), ",", elapsed, "s elapsed per epoch", flush=True)
        else:
            print("loss at epoch", epoch, ":", float(loss), ",", elapsed, "s elapsed per epoch", flush=True)
        start_time = time.time()

    print("finished at epoch", epoch, flush=True)

    if dev:
        print("final loss:", float(loss), "final dev_loss:", float(dev_loss), flush=True)
    else:
        print("final loss:", float(loss), flush=True)

    return float(dev_loss) if dev else float(loss), loss_function
