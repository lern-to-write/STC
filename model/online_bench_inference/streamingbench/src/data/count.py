import json
import csv
from collections import defaultdict
import argparse

def _get_model_answer(question, model_key):
    """兼容字符串和列表，统一返回去掉前后空格的答案字符"""
    raw = question.get(model_key)
    if raw is None:
        return None
    # 如果是 list 就取第一个元素，否则直接用
    ans = raw[0] if isinstance(raw, list) else raw
    return ans.strip()      # 去掉可能出现的 ' A' 或 'A ' 这类空格

def count(args):
    task = args.task
    src = args.src
    model = args.model

    # Load the JSON file
    with open(src, 'r') as file:
        data = json.load(file)

    # Initialize counters
    stats = defaultdict(lambda: defaultdict(int))

    # Process each entry in the JSON data
    total = 0

    if task == "sqa":
        for ques in data:
            for entry in ques:
                # if ques.index(entry) == 0:
                #     continue
                for question in entry["questions"]:
                    task_type = question["task_type"]
                    if model not in question or not question.get(model, None):
                        continue
                    model_answer = _get_model_answer(question, model).strip()
                    correct_answer = question["answer"]

                    if model_answer:
                        total += 1
                        stats[task_type]["total"] += 1
                        if correct_answer == model_answer:
                            stats[task_type]["correct"] += 1
    elif task == "proactive":
        for entry in data:
            for question in entry["questions"]:
                if model not in question:
                    continue
                ground_truth_timestamp = question["ground_truth_time_stamp"]
                ground_truth_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground_truth_timestamp.split(":"))))

                task_type = question["task_type"]
                model_answer = question.get(model, None).strip()
                history = model_answer["dialog_history"]
                last_time = history[-1]["time"]
                lase_answer = history[-1]["content"]

                if model_answer:
                    total += 1
                    stats[f'{task_type}']["total"] += 1
                    if -2 <= last_time - ground_truth_time <= 2:
                        stats[f'{task_type}']["time_correct"] += 1
                        if question["ground_truth_output"] in lase_answer:
                            stats[f'{task_type}']["answer_correct"] += 1
    else:
        for entry in data:
            for question in entry["questions"]:
                task_type = question["task_type"]
                if model not in question:
                    continue
                model_answer = _get_model_answer(question, model).strip()
                correct_answer = question["answer"]
                
                if model_answer:
                    total += 1
                    stats[task_type]["total"] += 1
                    stats["total"]["total"] += 1
                    if model_answer == correct_answer:
                        stats[task_type]["correct"] += 1
                        stats["total"]["correct"] += 1

    # Calculate accuracy for each task_type
    if task == "proactive":
        for task_type, counts in stats.items():
            counts["time_accuracy"] = counts["time_correct"] / counts["total"] if counts["total"] > 0 else 0
            counts["answer_accuracy"] = counts["answer_correct"] / counts["total"] if counts["total"] > 0 else 0
    else:
        for task_type, counts in stats.items():
            counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

    # Save results as a JSON file
    with open(f'{model}_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)

    # Save results as a CSV file
    with open(f'{model}_stats.csv', 'w', newline='') as csv_file:
        if task == "proactive":
            fieldnames = ["task_type", "total", "time_correct", "time_accuracy", "answer_correct", "answer_accuracy"]
        else:
            fieldnames = ["task_type", "total", "correct", "accuracy"]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        if task == "proactive":
            for task_type, counts in stats.items():
                writer.writerow({
                    "task_type": task_type,
                    "total": counts["total"],
                    "time_correct": counts["time_correct"],
                    "time_accuracy": counts["time_accuracy"],
                    "answer_correct": counts["answer_correct"],
                    "answer_accuracy": counts["answer_accuracy"]
                })
        else:
            for task_type, counts in stats.items():
                writer.writerow({
                    "task_type": task_type,
                    "total": counts["total"],
                    "correct": counts["correct"],
                    "accuracy": counts["accuracy"]
                })

    # --- 新增的打印代码 ---
    print(f"--- {model} 统计结果 ---")
    # 使用 json.dumps 格式化输出 stats 字典
    # ensure_ascii=False 可以确保中文等非 ASCII 字符正常显示
    print(json.dumps(stats, indent=4, ensure_ascii=False))
    print("------------------------")
    # --- 结束 ---

    print(f"{total} items have been statisticed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--src', type=str, help='Path to the data file')
    args = parser.parse_args()
    count(args)

if __name__ == "__main__":
    main()