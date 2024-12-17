# %%
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

outputs_dir = "evals/sparse_probing/results"
results = [f for f in os.listdir(outputs_dir) if f.startswith("gpt2-small")]
sizes = [int(re.search(r"(\d{3,})", f).group(0)) for f in results]

results = [x for _, x in sorted(zip(sizes, results))]
sizes.sort()

data = []
for r in results:
    with open(os.path.join(outputs_dir, r), "r") as f:
        data.append(json.load(f))

dataset_names = data[0]["eval_config"]["dataset_names"]
dataset_scores = defaultdict(list)

for sae in data:
    for result in sae["eval_result_details"]:
        dataset_name = result["dataset_name"][:-len("_results")]
        dataset_scores[dataset_name].append(result["sae_top_50_test_accuracy"])

num_ticks = len(sizes)
linear_ticks = range(num_ticks)

plt.figure(figsize=(10, 6))

for dataset_name, scores in dataset_scores.items():
    plt.plot(linear_ticks, scores, marker='o', label=dataset_name)

plt.xticks(ticks=linear_ticks, labels=[str(size) for size in sizes])

plt.xlabel('SAE Dictionary Size')
plt.ylabel('Accuracy')
plt.title('Sparse probing eval accuracy by SAE dictionary size')
legend = plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
legend.set_title("Benchmark Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()

# # %%
# import pandas as pd
# from IPython.display import display

# df = pd.DataFrame(dataset_scores, index=sizes)
# df = df.T
# styled_df = df.style.highlight_max(axis=1, color='green')
# display(styled_df)

# %%

outputs_dir = "evals/shift_and_tpp/results/tpp"
results = [f for f in os.listdir(outputs_dir) if f.startswith("gpt2-small")]
sizes = [int(re.search(r"(\d{3,})", f).group(0)) for f in results]

results = [x for _, x in sorted(zip(sizes, results))]
sizes.sort()

data = []
for r in results:
    with open(os.path.join(outputs_dir, r), "r") as f:
        data.append(json.load(f))

dataset_names = data[0]["eval_config"]["dataset_names"]
dataset_scores = defaultdict(list)

for sae in data:
    for result in sae["eval_result_details"]:
        dataset_name = result["dataset_name"][:-len("_results")]
        dataset_scores[dataset_name].append(result["tpp_threshold_50_total_metric"])

num_ticks = len(sizes)
linear_ticks = range(num_ticks)

plt.figure(figsize=(6, 4))

for dataset_name, scores in dataset_scores.items():
    plt.plot(linear_ticks, scores, marker='o', label=dataset_name)

plt.xticks(ticks=linear_ticks, labels=[str(size) for size in sizes])

plt.xlabel('SAE Dictionary Size')
plt.ylabel('Accuracy')
plt.title('TPP eval accuracy by SAE dictionary size')
legend = plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
legend.set_title("Benchmark Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()