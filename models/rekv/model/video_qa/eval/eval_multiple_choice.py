import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def calc_average_metric(results, save_dir, metric, vmin=None, vmax=None):
    if isinstance(results, list):
        average_metric = sum([item[metric] for item in results]) / len(results)
        print(f'#Samples: {len(results)}')
        print(f'Average {metric}: {average_metric:.2f}')

    elif isinstance(results, dict):
        average_recall = {}
        for key, value in results.items():
            recalls = [item[metric] for item in value]
            if len(value) > 0:
                average_recall[key] = (sum(recalls) / len(recalls))
            else:
                average_recall[key] = None

        df = pd.DataFrame.from_dict(average_recall, orient='index')
        df.index = pd.MultiIndex.from_tuples(df.index, names=['retrieve_size', 'chunk_size'])
        df = df.reset_index()
        df.columns = ['retrieve_size', 'chunk_size', 'value']
        heatmap_data = df.pivot(index='chunk_size', columns='retrieve_size', values='value')
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdPu", cbar_kws={'label': 'Value'}, 
                        xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        plt.title(f'Heatmap of Average {metric.capitalize()}')
        plt.xlabel('Retrieve Size')
        plt.ylabel('Chunk Size')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

        print(f'#Samples: {len(results[list(results.keys())[0]])}')
        print(average_recall)
        os.system(f"imgcat {os.path.join(save_dir, f'{metric}.png')}")
    else:
        raise ValueError(f"Invalid record type: {type(results)}")

    print(f'save_dir: {save_dir}')


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--results_path', type=str, default=None)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.results_path is not None:
    df = pd.read_csv(args.results_path)
    args.save_dir = os.path.dirname(args.results_path)
else:
    df = pd.read_csv(os.path.join(args.save_dir, 'results.csv'))

if 'retrieve_size' in df.columns:
    results = {}
    for _, row in df.iterrows():
        key = (row['retrieve_size'], row['chunk_size'])
        value = {col: row[col] for col in df.columns if col not in ['retrieve_size', 'chunk_size']}
        if key not in results:
            results[key] = []
        results[key].append(value)
else:
    results = df.to_dict(orient='records')

if 'recall' in df.columns:
    metrics = ['recall', 'precision', 'f1', 'qa_acc', 'acc_at_gqa']
else:
    metrics = ['qa_acc']

for metric in metrics:
    calc_average_metric(results, args.save_dir, metric)# vmin=0, vmax=100)

if 'pred_choice' in df.columns:
    n_errors = 0
    for _, row in df.iterrows():
        if row['pred_answer'][0] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            n_errors += 1
            if args.debug:
                print(f'Video: {row["video_id"]}, Question: {row["question"]}, GT: {row["correct_choice"]}, Pred: {row["pred_answer"]}')
    print(f'%Errors: {n_errors / len(df) * 100:.2f}')
