import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_absolute_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device being used:", device, flush=True)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return F.one_hot(y, num_classes)


def to_small_categorical(indices, num_categories):
    return torch.eye(num_categories, device=device)[indices]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_with_z_term(loss_fn, z_hat, y, seen=None, z_weight=1.0, eps=1e-6):
    y_clamp = torch.clamp(y, eps, 1.0 - eps)
    z = torch.log(y_clamp / (1-y_clamp))

    # y_view = y.view(-1, 1)
    if seen is not None:
        return torch.mean((loss_fn(z_hat, y).flatten() + z_weight * torch.square(z - z_hat).flatten()) * seen)
    else:
        return loss_fn(z_hat, y) + z_weight * torch.square(z - z_hat)


class DuoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, users_path, items_path, langs_path, skill_path, time_path, target_path, mask_path):
        self.users = np.load(data_dir+users_path)
        self.items = np.load(data_dir+items_path)
        self.langs = np.load(data_dir+langs_path)
        self.skills = np.load(data_dir+skill_path)
        self.timestamps = np.load(data_dir+time_path)
        self.targets = np.load(data_dir+target_path)
        self.mask = np.load(data_dir+mask_path)

        self.min_user = np.min(self.users)
        self.max_user = np.max(self.users)
        self.min_item = np.min(self.items)
        self.max_item = np.max(self.items)
        self.min_lang = np.min(self.langs)
        self.max_lang = np.max(self.langs)
        self.min_skill = np.min(self.skills)
        self.max_skill = np.max(self.skills)

    def update_user_limits(self, new_min, new_max):
        self.min_user = np.min([new_min, self.min_user])
        self.max_user = np.max([new_max, self.max_user])

    def update_item_limits(self, new_min, new_max):
        self.min_item = np.min([new_min, self.min_item])
        self.max_item = np.max([new_max, self.max_item])

    def update_lang_limits(self, new_min, new_max):
        self.min_lang = np.min([new_min, self.min_lang])
        self.max_lang = np.max([new_max, self.max_lang])

    def update_skill_limits(self, new_min, new_max):
        self.min_skill = np.min([new_min, self.min_skill])
        self.max_skill = np.max([new_max, self.max_skill])

    def __getitem__(self, idx):
        item = {"users": self.users[idx],
                "items": self.items[idx, :],
                "langs": self.langs[idx, :],
                "skills": self.skills[idx, :],
                "timestamps": self.timestamps[idx, :],
                "targets": self.targets[idx, :],
                "mask": self.mask[idx, :]}
        return item

    def __len__(self):
        return len(self.targets)

    def real_size(self):
        return np.sum(self.mask)


def process_batch(batch, z_weight):
    users = to_categorical(batch['users'].long().to(device).view(-1), max_user+1)
    items = to_categorical(batch['items'].long().to(device).view(-1), max_item+1)
    langs = to_categorical(batch['langs'].long().to(device).view(-1), max_lang+1)
    skills = batch['skills'].long().to(device)
    timestamps = batch['timestamps'].to(device)
    targets = batch['targets'].to(device)
    mask = batch['mask'].bool().to(device)
    loss, outputs, labels = model(users, items, langs, skills, timestamps, targets, mask, z_weight=z_weight)
    return labels, loss, outputs


def evaluate(data_loader, model, z_weight, return_outputs=False):
    model.eval()
    with torch.no_grad():
        all_labels = None
        all_preds = None
        all_loss = None
        total_count = 0
        for batch_i, batch in enumerate(data_loader):
            targets, loss, outputs = process_batch(batch, z_weight)
            total_count += len(targets)

            if all_labels is None:
                all_labels = targets
                all_preds = outputs
                all_loss = loss.sum()
            else:
                all_labels = torch.cat((all_labels, targets), dim=0)
                all_preds = torch.cat((all_preds, outputs), dim=0)
                all_loss += loss.sum()

        out_loss = float(all_loss) / float(total_count)
        all_labels = all_labels.cpu()
        all_preds = all_preds.cpu()
        auc = roc_auc_score(np.round(all_labels), all_preds)
        mae = mean_absolute_error(all_labels, all_preds)

        if return_outputs:
            return out_loss, auc, mae, all_labels, all_preds
        return out_loss, auc, mae


def save_model_and_predictions(model, all_labels, all_preds, out_modifier=""):
    # output predictions
    predictions_filename = out_filename
    if out_modifier != "":
        predictions_filename = predictions_filename.replace(".pt", ".predictions_" + out_modifier + ".csv")
    else:
        predictions_filename = predictions_filename.replace(".pt", ".predictions.csv")

    with open(predictions_filename, "w") as predictions_file:
        predictions_file.write("y,y_hat\n")
        for y, y_hat in zip(all_labels.tolist(), all_preds.tolist()):
            predictions_file.write(str(y) + "," + str(float(y_hat)) + "\n")

    model = model.cpu()

    model_filename = out_filename
    if out_modifier != "":
        model_filename = model_filename.replace(".pt", "." + out_modifier + ".pt")

    torch.save(model, model_filename)


class LogisticRegression(nn.Module):

    def __init__(self, num_features, num_skills, emb_size, emb_h1_size, emb_h2_size, history_len=200, dropout=0.0, embeddings=None, freeze=True, use_cuda=False, sum_log_relu=False):
        super(LogisticRegression, self).__init__()
        self.sum_log_relu = sum_log_relu

        self.emb_size = emb_size
        self.time_bins = torch.tensor([1, 3600, 24 * 3600, 7 * 24 * 3600], device=device)
        if embeddings is None:
            self.embeddings = nn.Embedding(num_skills, emb_size, padding_idx=0)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=freeze)

        self.num_skills = num_skills
        self.num_features = num_features
        self.history_len = history_len

        self.emb_h1 = nn.Linear((2 * emb_size) + self.time_bins.size()[0]+2, emb_h1_size)
        # self.emb_h1 = nn.Linear((2 * emb_size), emb_h1_size)
        self.drop1 = torch.nn.Dropout(dropout)
        self.emb_h2 = nn.Linear(emb_h1_size, emb_h2_size)
        self.drop2 = torch.nn.Dropout(dropout)
        self.emb_sim = nn.Linear(emb_h2_size, 1)
        self.emb_sim_activation = nn.Tanh()

        self.linear = nn.Linear(num_features + self.num_skills * 2, 1)
        self.use_cuda = use_cuda

        self.history_indices = torch.arange(history_len, device=device).repeat(history_len, 1)
        history_mask = (1 - torch.triu(torch.ones(history_len, history_len, device=device))) == 1

        self.history_indices_selected = self.history_indices[history_mask]
        self.adjusted_history_len = self.history_indices_selected.size()[0]

        self.skill_indices = torch.ones(history_len, history_len, dtype=torch.long, device=device) * torch.arange(history_len, dtype=torch.long, device=device).view(-1, 1)
        self.skill_indices = self.skill_indices[history_mask]

        self.sum_indices = torch.sum(torch.cat((self.history_indices * history_mask.long(), torch.arange(history_len, device=device).unsqueeze(0)), dim=0), dim=1)

    def categorize_time(self, times, skill_padding_mask, hist_skill_padding_mask):
        time_categories = torch.sum(times > self.time_bins, dim=1) + 1  # shift over one to make room for padding category
        time_categories[skill_padding_mask] = 0
        time_categories[hist_skill_padding_mask] = 0
        return to_small_categorical(time_categories.long(), self.time_bins.size()[0]+2)

    def compute_correct_total(self, skills, timestamps, targets):
        num_examples, history_len = skills.size()
        assert history_len == self.history_len

        skill_embeddings = self.embeddings(skills).float()
        hist_skill_embeddings = skill_embeddings[:, self.history_indices_selected].view(-1, self.emb_size)

        skill_embeddings = skill_embeddings[:, self.skill_indices].view(-1, self.emb_size)

        hist_times = timestamps[:, self.history_indices_selected].view(-1, 1)
        skill_times = timestamps[:, self.skill_indices].view(-1, 1)
        padding_mask = (skills == 0)
        delta_t = self.categorize_time(skill_times - hist_times, padding_mask[:, self.skill_indices].view(-1), padding_mask[:, self.history_indices_selected].view(-1))

        concat_emb = torch.cat((skill_embeddings, hist_skill_embeddings, delta_t), dim=1)

        hist_similarities = self.emb_sim_activation(self.emb_sim(self.drop2(F.relu(self.emb_h2(self.drop1(F.relu(self.emb_h1(concat_emb)))))))).view(num_examples, self.adjusted_history_len)

        total_correct_features = torch.zeros(num_examples * self.history_len, 2 * self.num_skills, dtype=torch.float32, device=device)

        flat_skills = skills.view(-1, 1)
        padding_mask = (flat_skills != 0).float()

        total_cumsum = torch.cumsum(hist_similarities, dim=1)
        zero_col = torch.zeros(num_examples, 1, device=device)
        total_cumsum = torch.cat((zero_col, total_cumsum), dim=1)
        ub = total_cumsum[:, self.sum_indices[1:]]
        lb = total_cumsum[:, self.sum_indices[:-1]]
        if self.sum_log_relu:
            total_correct_features.scatter_(1, flat_skills, torch.log1p(F.relu((ub - lb).view(-1, 1))) * padding_mask)
        else:
            total_correct_features.scatter_(1, flat_skills, (ub - lb).view(-1, 1) * padding_mask)

        correct_similarities = hist_similarities * targets[:, self.history_indices_selected]
        total_cumsum = torch.cumsum(correct_similarities.float(), dim=1)
        total_cumsum = torch.cat((zero_col, total_cumsum), dim=1)
        ub = total_cumsum[:, self.sum_indices[1:]]
        lb = total_cumsum[:, self.sum_indices[:-1]]
        if self.sum_log_relu:
            total_correct_features.scatter_(1, flat_skills+self.num_skills, torch.log1p(F.relu((ub - lb).view(-1, 1))) * padding_mask)
        else:
            total_correct_features.scatter_(1, flat_skills+self.num_skills, (ub - lb).view(-1, 1) * padding_mask)

        return total_correct_features

    def forward(self, users, items, langs, skills, timestamps, targets, mask, z_weight=0.0):
        num_examples, seqlen = skills.size()

        total_correct_intermediate = self.compute_correct_total(skills, timestamps, targets)

        concat_features = torch.cat((torch.repeat_interleave(users, seqlen, dim=0), items, langs, to_categorical(skills.view(-1), self.num_skills), total_correct_intermediate), dim=1)  # num_examples x (num_features)

        logits = self.linear(concat_features)
        loss_function = nn.BCEWithLogitsLoss(reduction='none')

        mask = mask.view(-1)
        preds = logits.view(-1)[mask]
        labels = targets.view(-1)[mask]

        loss = loss_with_z_term(loss_function, preds, labels, z_weight=z_weight)
        return loss, torch.sigmoid(preds), labels


if __name__ == "__main__":
    history_len = 200
    sum_log_relu = False
    print("history len:", history_len, flush=True)

    data_dir = "./data/duolingo_hlr/results"
    out_filename = data_dir + "das3h_neural_embeddings_2_"+str(history_len)+"_log1p="+str(sum_log_relu)+".pt"
    data_dir = "../dkt/"
    load_model = None

    modifier = "_" + str(history_len)  # "__continuous_continuous_wins_windows=[]_no_bias_all_word_embeddings_fastword"
    dev_dataset = DuoDataset(data_dir, "dev_users"+modifier+".npy", "dev_items"+modifier+".npy",
                               "dev_langs"+modifier+".npy", "dev_skills"+modifier+".npy", "dev_timestamp"+modifier+".npy",
                               "dev_p"+modifier+".npy", "dev_eval_mask"+modifier+".npy")
    print("loaded dev dataset:", dev_dataset.real_size(), flush=True)
    test_dataset = DuoDataset(data_dir, "test_users"+modifier+".npy", "test_items"+modifier+".npy",
                               "test_langs"+modifier+".npy", "test_skills"+modifier+".npy", "test_timestamp"+modifier+".npy",
                               "test_p"+modifier+".npy", "test_eval_mask"+modifier+".npy")
    print("loaded test dataset:", test_dataset.real_size(), flush=True)
    train_dataset = DuoDataset(data_dir, "train_users"+modifier+".npy", "train_items"+modifier+".npy",
                               "train_langs"+modifier+".npy", "train_skills"+modifier+".npy", "train_timestamp"+modifier+".npy",
                               "train_p"+modifier+".npy", "train_eval_mask"+modifier+".npy")
    print("loaded train dataset:", train_dataset.real_size(), flush=True)

    min_user = np.min([train_dataset.min_user, dev_dataset.min_user, test_dataset.min_user])
    max_user = np.max([train_dataset.max_user, dev_dataset.max_user, test_dataset.max_user])
    print("user:", min_user, max_user, flush=True)
    train_dataset.update_user_limits(min_user, max_user)
    dev_dataset.update_user_limits(min_user, max_user)
    test_dataset.update_user_limits(min_user, max_user)

    min_item = np.min([train_dataset.min_item, dev_dataset.min_item, test_dataset.min_item])
    max_item = np.max([train_dataset.max_item, dev_dataset.max_item, test_dataset.max_item])
    print("item:", min_item, max_item, flush=True)
    train_dataset.update_item_limits(min_item, max_item)
    dev_dataset.update_item_limits(min_item, max_item)
    test_dataset.update_item_limits(min_item, max_item)

    min_lang = np.min([train_dataset.min_lang, dev_dataset.min_lang, test_dataset.min_lang])
    max_lang = np.max([train_dataset.max_lang, dev_dataset.max_lang, test_dataset.max_lang])
    print("lang:", min_lang, max_lang, flush=True)
    train_dataset.update_lang_limits(min_lang, max_lang)
    dev_dataset.update_lang_limits(min_lang, max_lang)
    test_dataset.update_lang_limits(min_lang, max_lang)

    min_skill = np.min([train_dataset.min_skill, dev_dataset.min_skill, test_dataset.min_skill])
    max_skill = np.max([train_dataset.max_skill, dev_dataset.max_skill, test_dataset.max_skill])
    print("skill:", min_skill, max_skill, flush=True)
    train_dataset.update_skill_limits(min_skill, max_skill)
    dev_dataset.update_skill_limits(min_skill, max_skill)
    test_dataset.update_skill_limits(min_skill, max_skill)

    num_features = max_user+1 + max_item+1 + max_lang+1 + max_skill+1
    print("num precomputed features:", num_features, flush=True)

    embeddings_file = "./data/duolingo_hlr/embeddings_all_word_embeddings_fastword.npy" # None

    if embeddings_file is not None:
        print("Using pretrained embeddings.", flush=True)
        embeddings = torch.from_numpy(np.load(embeddings_file))
        print("embeddings shape:", np.shape(embeddings), flush=True)
        n_question, embed_dim = np.shape(embeddings)
    else:
        embeddings = None
        print("Training embeddings from scratch.", flush=True)
        n_question, embed_dim = 14003, 300

    real_batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=real_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=real_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=real_batch_size)

    train_batch_length = len(train_loader)
    print("train batches:", train_batch_length, flush=True)
    print("dev batches:", len(dev_loader), flush=True)
    print("test batches:", len(test_loader), flush=True)

    print("train size:", len(train_dataset), flush=True)
    print("dev size:", len(dev_dataset), flush=True)
    print("test size:", len(test_dataset), flush=True)

    freeze_pretrained = True
    print("Should freeze embeddings?:", freeze_pretrained, flush=True)
    grid_search = True
    print("grid search?:", grid_search, flush=True)

    # hyperparameters
    if grid_search:
        emb_h1_options = [512]
        emb_h2_options = [64]
        z_weight_options = [0.01]
        learning_rate_options = [5e-3]
        dropout_options = [0.1]
        batch_size_options = [real_batch_size*200]
    else:
        emb_h1_options = [128]
        emb_h2_options = [32]
        z_weight_options = [0.01]
        learning_rate_options = [5e-4]
        dropout_options = [0.2]
        batch_size_options = [real_batch_size]

    patience = 3

    num_epochs = 200000

    best_model_auc = None
    best_auc = None
    best_parameters_auc = None
    best_model_mae = None
    best_mae = None
    best_parameters_mae = None

    for emb_h1 in emb_h1_options:
        for emb_h2 in emb_h2_options:
            for z_weight in z_weight_options:
                for learning_rate in learning_rate_options:
                    for dropout in dropout_options:
                        for batch_size in batch_size_options:
                            current_parameters = (emb_h1, emb_h2, z_weight, learning_rate, dropout, batch_size)
                            print("-----new model parameters:", current_parameters, "-----", flush=True)

                            batches_needed_for_effective_size = int(batch_size / real_batch_size)
                            effective_num_batches = int(np.ceil(float(train_batch_length) / float(batches_needed_for_effective_size)))
                            print("effective number of training batches:", effective_num_batches, flush=True)

                            if load_model is not None:
                                model = torch.load(load_model)
                                print("loaded model:", load_model, flush=True)
                            else:
                                model = LogisticRegression(num_features, max_skill+1, 300, emb_h1, emb_h2, sum_log_relu=sum_log_relu, history_len=history_len, dropout=dropout, embeddings=embeddings, freeze=freeze_pretrained)
                            model.to(device)
                            print("model parameters:", count_parameters(model), flush=True)

                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                            # start training
                            best_loss = None
                            patience_used = 0
                            start_time = time.time()

                            for epoch in range(num_epochs):
                                batch_count = 0
                                batch_size_counter = 0
                                model.train()
                                for batch_i, batch in enumerate(train_loader):
                                    _, loss, outputs = process_batch(batch, z_weight)
                                    loss = loss.mean()
                                    loss.backward()

                                    batch_size_counter += 1
                                    if batch_size_counter == batches_needed_for_effective_size or batch_i == train_batch_length - 1:
                                        optimizer.step()
                                        optimizer.zero_grad()
                                        batch_size_counter = 0

                                        batch_count += 1

                                        current_effective_batch = int((batch_i + 1) / batches_needed_for_effective_size)

                                        if current_effective_batch % 1 == 0:
                                            print("batch", current_effective_batch, "/", effective_num_batches, ", loss:", float(loss), ", average time per batch:", (time.time() - start_time) / batch_count, flush=True)
                                            start_time = time.time()
                                            batch_count = 0

                                dev_loss, dev_auc, dev_mae, all_labels, all_preds = evaluate(dev_loader, model, z_weight, return_outputs=True)

                                dev_loss = float(dev_loss)
                                print("epoch", epoch, "dev loss:", dev_loss, "patience used:", patience_used, "/", patience, "dev auc:", dev_auc, "dev mae:", dev_mae, flush=True)

                                if best_loss is None or dev_loss < best_loss:
                                    patience_used = 0
                                    best_loss = dev_loss
                                else:
                                    patience_used += 1
                                    if patience_used >= patience:
                                        break  # end training

                            test_loss, test_auc, test_mae, all_labels, all_preds = evaluate(test_loader, model, z_weight, return_outputs=True)
                            print("test loss:", test_loss, "test auc:", test_auc, "test mae:", test_mae, "parameters:", current_parameters, flush=True)

                            if best_auc is None or dev_auc > best_auc:
                                best_auc = dev_auc
                                best_model_auc = copy.deepcopy(model.cpu())
                                best_parameters_auc = current_parameters

                            if best_mae is None or dev_mae < best_mae:
                                best_mae = dev_mae
                                best_model_mae = copy.deepcopy(model.cpu())
                                best_parameters_mae = current_parameters

    if grid_search:
        print("---best model AUC:", best_auc, "parameters:", best_parameters_auc, "---", flush=True)
        model = best_model_auc
        model = model.to(device)

        test_loss, test_auc, test_mae, all_labels, all_preds = evaluate(test_loader, model, best_parameters_auc[2], return_outputs=True)
        print("test loss:", test_loss, "test auc:", test_auc, "test mae:", test_mae, flush=True)

        save_model_and_predictions(model, all_labels, all_preds, out_modifier="best_auc_"+str(best_parameters_auc).replace(" ", ""))

        print("---best model MAE:", best_mae, "parameters:", best_parameters_mae, "---", flush=True)
        model = best_model_mae
        model = model.to(device)

        test_loss, test_auc, test_mae, all_labels, all_preds = evaluate(test_loader, model, best_parameters_mae[2], return_outputs=True)
        print("test loss:", test_loss, "test auc:", test_auc, "test mae:", test_mae, flush=True)

        save_model_and_predictions(model, all_labels, all_preds, out_modifier="best_mae_"+str(best_parameters_mae).replace(" ", ""))
