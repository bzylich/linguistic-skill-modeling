import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run DAS3H.')
parser.add_argument('--data', type=str, nargs='?', default="../../data/learning_traces.13m.csv")
parser.add_argument('--emb_file', type=str, nargs='?', default='../all_word_embeddings_fastword.csv')
parser.add_argument('--num_bins', type=int, nargs='?', default=5)

options = parser.parse_args()

init_time = None

last_time = {}
all_delta_ts = []

with open(options.data, encoding="utf-8") as data_file:
    data_file.readline()
    for i, line in enumerate(data_file):
        _, timestamp, _, user_id, _, _, _, _, _, _, _, _ = line.strip().split(",")

        timestamp = float(timestamp)
        if init_time is None:
            init_time = timestamp
        timestamp -= init_time
        assert timestamp >= 0

        if user_id in last_time:
            delta_t = timestamp - last_time[user_id]

            # assert delta_t >= 0
            # if delta_t != 0:
            #     all_delta_ts.append(delta_t)
            all_delta_ts.append(delta_t)

        last_time[user_id] = timestamp

        if i % 100000 == 0:
            print(i, "processed", flush=True)

# bins = [0, 1, 60*10, 60*60, 60*60*24, np.max(all_delta_ts)]
print(np.max(all_delta_ts))
bins = [0, 1, 802.0, 4604.0, 81849.0, 1015617.0]

# frac = 100.0/float(options.num_bins)
# bins = [0]
# for i in range(options.num_bins):
#     if i < options.num_bins - 1:
#         bins.append(np.percentile(all_delta_ts, frac * (i+1)))
# bins.append(np.max(all_delta_ts))

print("bins:", bins, flush=True)
plt.xscale('log')
plt.yscale('log')
plt.hist(all_delta_ts, bins=bins)
plt.show()
