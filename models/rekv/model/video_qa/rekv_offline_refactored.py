"""重构的离线推理 - 简洁优雅的实现"""
import torch
from logzero import logger
from .base_refactored import BaseVQA
import torch.distributed as dist


class ReKVOfflineVQA(BaseVQA):
    """离线视频问答 - 所有函数<15行"""
    
    def answer_single(self, qa_pair, video_id):
        """回答单个问题"""
        if 'choices' in qa_pair:
            return self._multiple_choice_qa(qa_pair, video_id)
        return self._open_qa(qa_pair, video_id)
    
    def _open_qa(self, qa_pair, video_id):
        """开放式问答"""
        question = qa_pair['question']
        prompt = self.format_openqa_prompt(question)
        pred = self.model.question_answering(
            {"question": question, "prompt": prompt},
            max_new_tokens=1024
        )
        return self._format_open_result(pred, qa_pair, video_id)
    
    def _multiple_choice_qa(self, qa_pair, video_id):
        """多选题问答"""
        question = qa_pair['question']
        choices = qa_pair['choices']
        prompt = self.format_mcqa_prompt(question, choices)
        
        pred = self.model.question_answering(
            {"question": question, "prompt": prompt},
            max_new_tokens=16
        )
        return self._format_mc_result(pred, qa_pair, video_id)
    
    def _format_open_result(self, pred, qa_pair, video_id):
        """格式化开放式问答结果"""
        return {
            'video_id': video_id,
            'question': qa_pair['question'],
            'answer': qa_pair.get('answer'),
            'pred_answer': pred.replace('\n', ''),
        }
    
    def _format_mc_result(self, pred, qa_pair, video_id):
        """格式化多选题结果"""
        pred_choice = self.extract_choice(pred)
        correct_choice = self._get_correct_choice(qa_pair)
        
        return {
            'video_id': video_id,
            'question': qa_pair['question'],
            'choices': qa_pair['choices'],
            'answer': qa_pair.get('answer'),
            'correct_choice': correct_choice,
            'pred_answer': pred.replace('\n', ''),
            'pred_choice': pred_choice,
            'qa_acc': float(pred_choice == correct_choice) * 100,
        }
    
    def _get_correct_choice(self, qa_pair):
        """获取正确选项"""
        answer = qa_pair.get('answer')
        if answer is None:
            return self.choice_letters[0]
        
        choices = qa_pair['choices']
        try:
            idx = choices.index(answer)
            return self.choice_letters[idx]
        except ValueError:
            logger.warning(f"Answer not in choices: {answer}")
            return self.choice_letters[0]

