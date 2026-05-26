#!/usr/bin/env python3
"""
视频问答(Video QA)评测脚本
用于分析模型输出结果并生成详细的评测报告

使用方法：
    python evaluate_results.py --result_file results.csv
    python evaluate_results.py --result_dir results/batch_20251010_155403
    
"""

import os
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class VideoQAEvaluator:
    """视频问答评测器"""
    
    # 答案索引到选项字母的映射
    INDEX_TO_CHOICE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    CHOICE_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    def __init__(self, result_file: str):
        """
        初始化评测器
        
        Args:
            result_file: 结果CSV文件路径
        """
        self.result_file = result_file
        self.df = None
        self.metrics = {}
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载结果数据"""
        print(f"📂 加载结果文件: {self.result_file}")
        
        if not os.path.exists(self.result_file):
            raise FileNotFoundError(f"结果文件不存在: {self.result_file}")
        
        self.df = pd.read_csv(self.result_file)
        print(f"✅ 成功加载 {len(self.df)} 条数据")
        
        # 检查必需的列
        required_columns = ['video_id', 'question', 'answer', 'pred_choice', 'qa_acc']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        
        print(f"📊 数据列: {list(self.df.columns)}")
    
    def calculate_metrics(self) -> Dict:
        """
        计算各种评测指标
        
        Returns:
            包含所有指标的字典
        """
        print("\n" + "="*60)
        print("🔍 计算评测指标...")
        print("="*60)
        
        # 基础统计
        total_samples = len(self.df)
        correct_samples = (self.df['qa_acc'] == 1.0).sum()
        accuracy = self.df['qa_acc'].mean() * 100
        
        self.metrics['basic'] = {
            'total_samples': total_samples,
            'correct_samples': int(correct_samples),
            'wrong_samples': int(total_samples - correct_samples),
            'accuracy': accuracy,
            'error_rate': 100 - accuracy
        }
        
        # 按视频统计（如果有多个视频）
        video_stats = self.df.groupby('video_id').agg({
            'qa_acc': ['count', 'sum', 'mean']
        }).round(4)
        
        self.metrics['per_video'] = video_stats
        
        # 答案分布分析 - 转换为字母
        if 'answer' in self.df.columns:
            # 将索引转换为字母
            answer_letters = self.df['answer'].map(self.INDEX_TO_CHOICE)
            answer_dist = answer_letters.value_counts().sort_index()
            self.metrics['answer_distribution'] = answer_dist.to_dict()
        
        # 预测答案分布分析
        if 'pred_choice' in self.df.columns:
            pred_dist = self.df['pred_choice'].value_counts().sort_index()
            self.metrics['pred_distribution'] = pred_dist.to_dict()
        
        # 混淆矩阵 - 正确答案 vs 预测答案
        if 'answer' in self.df.columns and 'pred_choice' in self.df.columns:
            confusion = pd.crosstab(
                self.df['answer'].map(self.INDEX_TO_CHOICE),
                self.df['pred_choice'],
                rownames=['Ground Truth'],
                colnames=['Predicted']
            )
            self.metrics['confusion_matrix'] = confusion.to_dict()
        
        # 配置参数统计
        if 'retrieve_size' in self.df.columns:
            self.metrics['config'] = {
                'retrieve_size': self.df['retrieve_size'].iloc[0] if len(self.df) > 0 else None,
                'chunk_size': self.df['chunk_size'].iloc[0] if 'chunk_size' in self.df.columns and len(self.df) > 0 else None
            }
        
        return self.metrics
    
    def print_summary(self):
        """打印评测摘要"""
        if not self.metrics:
            self.calculate_metrics()
        
        basic = self.metrics['basic']
        
        print("\n" + "="*60)
        print("📊 评测结果摘要")
        print("="*60)
        print(f"总样本数:        {basic['total_samples']}")
        print(f"正确数量:        {basic['correct_samples']} ✅")
        print(f"错误数量:        {basic['wrong_samples']} ❌")
        print(f"准确率:          {basic['accuracy']:.2f}%")
        print(f"错误率:          {basic['error_rate']:.2f}%")
        print("="*60)
        
        # 配置信息
        if 'config' in self.metrics and self.metrics['config']['retrieve_size']:
            print(f"\n📝 配置参数:")
            print(f"检索大小 (retrieve_size): {self.metrics['config']['retrieve_size']}")
            if self.metrics['config']['chunk_size']:
                print(f"块大小 (chunk_size): {self.metrics['config']['chunk_size']}")
        
        # 答案分布
        if 'answer_distribution' in self.metrics:
            print(f"\n📈 正确答案分布:")
            for ans, count in sorted(self.metrics['answer_distribution'].items()):
                percentage = (count / basic['total_samples']) * 100
                print(f"  选项 {ans}: {count} 次 ({percentage:.1f}%)")
        
        # 预测分布
        if 'pred_distribution' in self.metrics:
            print(f"\n🎯 模型预测分布:")
            for choice, count in sorted(self.metrics['pred_distribution'].items()):
                percentage = (count / basic['total_samples']) * 100
                print(f"  选项 {choice}: {count} 次 ({percentage:.1f}%)")
        
        # 混淆矩阵
        if 'confusion_matrix' in self.metrics:
            print(f"\n🔀 混淆矩阵 (Ground Truth vs Predicted):")
            confusion_df = pd.DataFrame(self.metrics['confusion_matrix']).fillna(0).astype(int)
            print(confusion_df.to_string())
    
    def analyze_errors(self, top_n: int = 10) -> pd.DataFrame:
        """
        分析错误样本
        
        Args:
            top_n: 显示前N个错误样本
            
        Returns:
            错误样本的DataFrame
        """
        print(f"\n🔎 分析错误样本 (显示前{top_n}个)...")
        print("="*60)
        
        # 获取错误样本
        error_df = self.df[self.df['qa_acc'] == 0.0].copy()
        
        if len(error_df) == 0:
            print("🎉 没有错误样本！所有预测都正确！")
            return error_df
        
        print(f"总错误数: {len(error_df)}\n")
        
        # 显示前N个错误
        for i, (idx, row) in enumerate(error_df.head(top_n).iterrows(), 1):
            print(f"错误样本 #{i}")
            print(f"  视频ID: {row['video_id']}")
            print(f"  问题: {row['question'][:100]}...")  # 只显示前100个字符
            
            # 转换索引为字母
            correct_letter = self.INDEX_TO_CHOICE.get(row['answer'], '?')
            correct_text = str(row.get('correct_choice', 'N/A'))
            
            print(f"  正确答案: {correct_letter}) {correct_text[:80]}...")
            
            # 预测答案
            pred_letter = str(row.get('pred_choice', '?'))
            pred_text = str(row.get('pred_answer', 'N/A'))
            print(f"  模型预测: {pred_letter}) {pred_text[:80]}...")
            print()
        
        # 错误分析统计
        print("\n📊 错误分析统计:")
        
        # 统计每个正确答案的错误率
        if 'answer' in error_df.columns:
            error_by_answer = error_df['answer'].map(self.INDEX_TO_CHOICE).value_counts().sort_index()
            total_by_answer = self.df['answer'].map(self.INDEX_TO_CHOICE).value_counts().sort_index()
            
            print("\n各选项的错误分布:")
            for choice in sorted(set(list(error_by_answer.index) + list(total_by_answer.index))):
                errors = error_by_answer.get(choice, 0)
                total = total_by_answer.get(choice, 0)
                error_rate = (errors / total * 100) if total > 0 else 0
                print(f"  正确答案为{choice}: {errors}/{total} 错误 ({error_rate:.1f}%)")
        
        # 统计最常见的错误预测
        if 'pred_choice' in error_df.columns:
            print("\n错误样本中最常见的预测:")
            pred_counts = error_df['pred_choice'].value_counts().head(5)
            for pred, count in pred_counts.items():
                percentage = (count / len(error_df)) * 100
                print(f"  预测{pred}: {count} 次 ({percentage:.1f}%)")
        
        return error_df
    
    def save_detailed_report(self, output_dir: Optional[str] = None):
        """
        保存详细报告
        
        Args:
            output_dir: 输出目录，默认与结果文件同目录
        """
        if output_dir is None:
            output_dir = os.path.dirname(self.result_file)
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存JSON格式的指标
        metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.json')
        
        # 转换不可序列化的对象
        json_metrics = {}
        for key, value in self.metrics.items():
            if key == 'per_video':
                # 将多级索引的 DataFrame 转换为可序列化的格式
                per_video_dict = {}
                for video_id, row in value.iterrows():
                    per_video_dict[str(video_id)] = {
                        'count': int(row[('qa_acc', 'count')]),
                        'correct': int(row[('qa_acc', 'sum')]),
                        'accuracy': float(row[('qa_acc', 'mean')])
                    }
                json_metrics[key] = per_video_dict
            elif key == 'confusion_matrix':
                # 确保混淆矩阵的键都是字符串
                if isinstance(value, dict):
                    json_metrics[key] = {str(k): v for k, v in value.items()}
                else:
                    json_metrics[key] = value
            else:
                json_metrics[key] = value
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"💾 指标已保存: {metrics_file}")
        
        # 2. 保存Markdown格式的报告
        report_file = os.path.join(output_dir, f'evaluation_report_{timestamp}.md')
        self._generate_markdown_report(report_file)
        print(f"📝 报告已保存: {report_file}")
        
        # 3. 保存错误样本
        error_df = self.df[self.df['qa_acc'] == 0.0]
        if len(error_df) > 0:
            error_file = os.path.join(output_dir, f'error_samples_{timestamp}.csv')
            error_df.to_csv(error_file, index=False)
            print(f"❌ 错误样本已保存: {error_file}")
    def _generate_markdown_report(self, output_file: str):
        """生成Markdown格式的报告"""
        basic = self.metrics['basic']
        
        md_content = f"""# 📊 视频问答评测报告

    **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
    **结果文件**: `{self.result_file}`

    ---

    ## 1. 总体评测结果

    | 指标 | 数值 |
    |------|------|
    | 总样本数 | {basic['total_samples']} |
    | 正确数量 | {basic['correct_samples']} ✅ |
    | 错误数量 | {basic['wrong_samples']} ❌ |
    | **准确率** | **{basic['accuracy']:.2f}%** |
    | 错误率 | {basic['error_rate']:.2f}% |

    ---

    ## 2. 配置参数

    """
        
        if 'config' in self.metrics:
            config = self.metrics['config']
            md_content += f"- **检索大小 (retrieve_size)**: {config.get('retrieve_size', 'N/A')}\n"
            md_content += f"- **块大小 (chunk_size)**: {config.get('chunk_size', 'N/A')}\n"
        
        md_content += "\n---\n\n## 3. 答案分布分析\n\n"
        
        # 正确答案分布
        if 'answer_distribution' in self.metrics:
            md_content += "### 3.1 正确答案分布\n\n"
            md_content += "| 选项 | 出现次数 | 占比 |\n"
            md_content += "|------|---------|------|\n"
            
            for ans, count in sorted(self.metrics['answer_distribution'].items()):
                percentage = (count / basic['total_samples']) * 100
                md_content += f"| {ans} | {count} | {percentage:.1f}% |\n"
        
        # 预测分布
        if 'pred_distribution' in self.metrics:
            md_content += "\n### 3.2 模型预测分布\n\n"
            md_content += "| 预测选项 | 次数 | 占比 |\n"
            md_content += "|---------|------|------|\n"
            
            for choice, count in sorted(self.metrics['pred_distribution'].items()):
                percentage = (count / basic['total_samples']) * 100
                md_content += f"| {choice} | {count} | {percentage:.1f}% |\n"
        
        # 混淆矩阵
        if 'confusion_matrix' in self.metrics:
            md_content += "\n### 3.3 混淆矩阵 (Ground Truth vs Predicted)\n\n"
            try:
                confusion_df = pd.DataFrame(self.metrics['confusion_matrix']).fillna(0).astype(int)
                # 确保行和列都按字母顺序排列
                all_choices = sorted(set(list(confusion_df.index) + list(confusion_df.columns)))
                confusion_df = confusion_df.reindex(index=all_choices, columns=all_choices, fill_value=0)
                md_content += confusion_df.to_markdown() + "\n"
            except Exception as e:
                md_content += f"无法生成混淆矩阵: {e}\n"
        
        # 每个视频的统计
        if 'per_video' in self.metrics and len(self.metrics['per_video']) > 1:
            md_content += "\n---\n\n## 4. 按视频统计\n\n"
            md_content += "| 视频ID | 样本数 | 正确数 | 准确率 |\n"
            md_content += "|--------|--------|--------|--------|\n"
            
            for video_id, row in self.metrics['per_video'].iterrows():
                count = int(row[('qa_acc', 'count')])
                correct = int(row[('qa_acc', 'sum')])
                acc = row[('qa_acc', 'mean')] * 100
                video_id_short = str(video_id)[:30] + "..." if len(str(video_id)) > 30 else str(video_id)
                md_content += f"| {video_id_short} | {count} | {correct} | {acc:.2f}% |\n"
        
        md_content += "\n---\n\n## 5. 性能分析\n\n"
        
        if basic['accuracy'] >= 80:
            md_content += "✅ **优秀**: 模型表现出色，准确率超过80%\n"
        elif basic['accuracy'] >= 60:
            md_content += "⚠️ **良好**: 模型表现尚可，但仍有改进空间\n"
        elif basic['accuracy'] >= 40:
            md_content += "⚠️ **一般**: 模型表现中等，需要优化\n"
        elif basic['accuracy'] > 0:
            md_content += "❌ **较差**: 模型表现较差，需要重大改进\n"
        else:
            md_content += "🚨 **完全失败**: 所有预测都错误！请检查:\n"
            md_content += "   - 数据格式是否正确\n"
            md_content += "   - 答案索引是否对齐\n"
            md_content += "   - 模型输出是否有效\n"
        
        md_content += f"\n### 改进建议\n\n"
        
        if basic['accuracy'] == 0:
            md_content += "🚨 **紧急**: 模型完全没有预测正确，请立即检查:\n"
            md_content += "1. 检查答案格式和索引是否正确对齐\n"
            md_content += "2. 验证模型输出格式是否符合预期\n"
            md_content += "3. 检查数据预处理流程是否有误\n"
            md_content += "4. 确认评测脚本的逻辑是否正确\n"
        elif basic['error_rate'] > 50:
            md_content += "1. 检查模型架构和训练数据质量\n"
            md_content += "2. 考虑增加训练数据或改进数据增强策略\n"
            md_content += "3. 调整超参数或训练策略\n"
        elif basic['error_rate'] > 20:
            md_content += "1. 分析错误样本，找出模型的薄弱环节\n"
            md_content += "2. 考虑针对性地改进模型或数据\n"
            md_content += "3. 可以尝试集成学习或模型融合\n"
        else:
            md_content += "1. 继续保持当前策略\n"
            md_content += "2. 可以尝试更复杂的场景或数据集\n"
            md_content += "3. 考虑模型压缩和效率优化\n"
        
        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def visualize_results(self, output_dir: Optional[str] = None, show: bool = False):
        """
        可视化评测结果
        
        Args:
            output_dir: 输出目录
            show: 是否显示图表
        """
        if output_dir is None:
            output_dir = os.path.dirname(self.result_file)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Video QA Evaluation Results', fontsize=16, fontweight='bold')
        
        basic = self.metrics['basic']
        
        # 1. 准确率饼图
        ax1 = axes[0, 0]
        sizes = [basic['correct_samples'], basic['wrong_samples']]
        labels = ['Correct', 'Wrong']
        colors = ['#4CAF50', '#F44336']
        explode = (0.1, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title(f"Accuracy: {basic['accuracy']:.2f}%")
        
        # 2. 答案分布柱状图
        ax2 = axes[0, 1]
        if 'answer_distribution' in self.metrics:
            ans_dist = self.metrics['answer_distribution']
            choices = sorted(ans_dist.keys())
            counts = [ans_dist[c] for c in choices]
            ax2.bar(choices, counts, color='#2196F3')
            ax2.set_xlabel('Answer Choice')
            ax2.set_ylabel('Count')
            ax2.set_title('Ground Truth Answer Distribution')
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. 预测分布柱状图
        ax3 = axes[1, 0]
        if 'pred_distribution' in self.metrics:
            pred_dist = self.metrics['pred_distribution']
            choices = sorted(pred_dist.keys())
            counts = [pred_dist[c] for c in choices]
            ax3.bar(choices, counts, color='#FF9800')
            ax3.set_xlabel('Predicted Choice')
            ax3.set_ylabel('Count')
            ax3.set_title('Model Prediction Distribution')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. 统计信息文本
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
Total Samples: {basic['total_samples']}
Correct: {basic['correct_samples']}
Wrong: {basic['wrong_samples']}
Accuracy: {basic['accuracy']:.2f}%
Error Rate: {basic['error_rate']:.2f}%
        """
        
        if 'config' in self.metrics:
            config = self.metrics['config']
            stats_text += f"""
Retrieve Size: {config.get('retrieve_size', 'N/A')}
Chunk Size: {config.get('chunk_size', 'N/A')}
            """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(output_dir, f'evaluation_plot_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存: {plot_file}")
        
        if show:
            plt.show()
        else:
            plt.close()


def compare_experiments(result_files: List[str], output_dir: str):
    """
    比较多个实验结果
    
    Args:
        result_files: 结果文件列表
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("🔄 比较多个实验结果...")
    print("="*60)
    
    results = []
    
    for result_file in result_files:
        try:
            evaluator = VideoQAEvaluator(result_file)
            evaluator.calculate_metrics()
            
            exp_name = Path(result_file).parent.name
            results.append({
                'experiment': exp_name,
                'file': result_file,
                'metrics': evaluator.metrics['basic']
            })
        except Exception as e:
            print(f"⚠️ 无法加载 {result_file}: {e}")
    
    if not results:
        print("❌ 没有有效的结果文件")
        return
    
    # 创建对比表
    comparison_df = pd.DataFrame([
        {
            'Experiment': r['experiment'],
            'Total': r['metrics']['total_samples'],
            'Correct': r['metrics']['correct_samples'],
            'Wrong': r['metrics']['wrong_samples'],
            'Accuracy (%)': f"{r['metrics']['accuracy']:.2f}",
            'Error Rate (%)': f"{r['metrics']['error_rate']:.2f}"
        }
        for r in results
    ])
    
    print("\n📊 实验对比:")
    print(comparison_df.to_string(index=False))
    
    # 保存对比报告
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    comparison_file = os.path.join(output_dir, f'comparison_{timestamp}.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 对比结果已保存: {comparison_file}")
    
    # 生成对比图表
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        experiments = [r['experiment'] for r in results]
        accuracies = [r['metrics']['accuracy'] for r in results]
        
        bars = ax.bar(experiments, accuracies, color='#4CAF50')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Experiment Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'comparison_plot_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 对比图表已保存: {plot_file}")
        plt.close()


def evaluate_batch_results(batch_dir: str):
    """
    评测批量实验结果
    
    Args:
        batch_dir: 批量实验结果目录（包含多个子目录）
    """
    print(f"\n🔍 扫描批量实验目录: {batch_dir}")
    
    result_files = []
    
    # 递归查找所有results.csv文件
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            if file == 'results.csv':
                result_file = os.path.join(root, file)
                result_files.append(result_file)
    
    if not result_files:
        print(f"❌ 在 {batch_dir} 中未找到results.csv文件")
        return
    
    print(f"✅ 找到 {len(result_files)} 个结果文件\n")
    
    # 评测每个实验
    for result_file in result_files:
        print("\n" + "="*80)
        exp_dir = os.path.dirname(result_file)
        exp_name = os.path.basename(exp_dir)
        print(f"📊 评测实验: {exp_name}")
        print("="*80)
        
        try:
            evaluator = VideoQAEvaluator(result_file)
            evaluator.calculate_metrics()
            evaluator.print_summary()
            evaluator.analyze_errors(top_n=3)
            evaluator.save_detailed_report(output_dir=exp_dir)
            # evaluator.visualize_results(output_dir=exp_dir)  # 可选：生成图表
        except Exception as e:
            print(f"❌ 评测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比所有实验
    if len(result_files) > 1:
        compare_experiments(result_files, batch_dir)


def main():
    parser = argparse.ArgumentParser(description='视频问答评测脚本')
    parser.add_argument('--result_file', type=str, help='单个结果CSV文件路径')
    parser.add_argument('--result_dir', type=str, help='批量结果目录路径')
    parser.add_argument('--output_dir', type=str, help='输出目录（可选）')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    parser.add_argument('--show_plot', action='store_true', help='显示图表')
    parser.add_argument('--top_errors', type=int, default=10, help='显示的错误样本数量')
    
    args = parser.parse_args()
    
    if args.result_dir:
        # 批量评测模式
        evaluate_batch_results(args.result_dir)
    
    elif args.result_file:
        # 单文件评测模式
        print("="*60)
        print("🚀 开始评测...")
        print("="*60)
        
        evaluator = VideoQAEvaluator(args.result_file)
        evaluator.calculate_metrics()
        evaluator.print_summary()
        evaluator.analyze_errors(top_n=args.top_errors)
        
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.result_file)
        evaluator.save_detailed_report(output_dir=output_dir)
        
        if args.visualize:
            evaluator.visualize_results(output_dir=output_dir, show=args.show_plot)
        
        print("\n" + "="*60)
        print("✅ 评测完成!")
        print("="*60)
    
    else:
        print("❌ 请指定 --result_file 或 --result_dir 参数")
        parser.print_help()


if __name__ == "__main__":
    main()