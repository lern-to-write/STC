import os
import json
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--results_path', type=str, default=None)
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
    results = results[(df['retrieve_size'].max(), 1)]
else:
    results = df.to_dict(orient='records')

submission = []
for r in results:
    if r['pred_choice'] not in ['A', 'B', 'C', 'D', 'E']:
        print(f"r['pred_choice']: {r['pred_choice']}")
        r['pred_choice'] = 'A'
    answer = ord(r['pred_choice']) - ord('A')
    submission.append({
        'q_uid': r['video_id'],
        'answer': answer
    })
submission = pd.DataFrame(submission)
submission.to_csv(os.path.join(args.save_dir, 'submission.csv'), index=False)
