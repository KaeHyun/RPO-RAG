import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable

import json
with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}

import re 
import string
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

class PromptBuilder(object):
    BASE_INSTRUCTION= """"""
    CUSTOM_INSTRUCTION = """Based on the Answer-Centered-Paths, please answer the given question.  
            The Answer-Centered-Paths helps you to step-by-step reasoning to answer the question.  

            Let's think step by step. Return the most possible answers based on the given paths by listing each answer on a separate line.  
            Please keep the answer as simple as possible and return all the possible answers as a list."""

    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    EACH_LINE = """ Please return each answer in a new line."""
    def __init__(self, prompt_path, encrypt=False, add_rule = False, use_true = False, cot = False, explain = False, use_random = False, each_line = False, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line

        self.encrypt=encrypt
        
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        
        entities = question_dict['q_entity']
        skip_ents = []
        
        graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)

        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        
        if not question.endswith('?'):
            question += '?'
        
        meta_lines = []
        if self.add_rule:
            entities = question_dict['q_entity']
            skip_ents = []
            graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)

            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']

            if len(rules) > 0:
                content_lines = []
                # 문자열 규칙(Reasoning Paths or Answer-Centered Paths)일 때
                if isinstance(rules[0], str) and ("Reasoning Paths:" in rules[0] or "Answer-Centered Paths:" in rules[0]):
                    # 메타는 ACC 헤더만 고정 포함
                    meta_lines = ["Answer-Centered Paths:"]
                    for rule in rules:
                        # 어느 토큰이든 뒤 본문만 가져오기
                        token = "Answer-Centered Paths:" if "Answer-Centered Paths:" in rule else "Reasoning Paths:"
                        body = rule.split(token, 1)[1].strip()

                        # 본문에서 섹션/경로 라인만 컨텐츠로 수집
                        for ln in body.splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            if ln.startswith("[") or ln.startswith("<") or "->" in ln:
                                content_lines.append(ln)
                else:
                    # 리스트 규칙 포맷일 때: 그래프 탐색 결과를 문자열로
                    reasoning_paths = self.apply_rules(graph, rules, entities)
                    content_lines = [utils.path_to_string(p) for p in reasoning_paths]
                    # 메타는 ACC 헤더만 고정 포함
                    meta_lines = ["Answer-Centered Paths:"]
            else:
                meta_lines = []
                content_lines = []
        
        input = self.QUESTION.format(question = question)
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule:
            other_prompt = self.prompt_template.format(
                instruction = instruction,
                input = self.GRAPH_CONTEXT.format(context = "") + input
            )
            # meta_lines / content_lines 전달
            context = self.check_prompt_length(other_prompt, meta_lines, content_lines, self.maximun_token)
            input = self.GRAPH_CONTEXT.format(context = context) + input
        
        input = self.prompt_template.format(instruction = instruction, input = input)
            
        return input
    
    def check_prompt_length(self, prompt, meta_lines, content_lines, maximun_token):
        """
        meta_lines: 반드시 포함(헤더/라벨)
        content_lines: 길이 컷의 대상(섹션/경로)
        """
        keep = []
        # 메타 먼저 고정 삽입
        base = prompt + "\n".join(meta_lines)
        if self.tokenize(base) >= maximun_token:
            # 정말 토큰 한도가 너무 작다면 메타만 최대한 잘라 포함 (안전장치)
            trimmed_meta = []
            for m in meta_lines:
                candidate = prompt + "\n".join(trimmed_meta + [m])
                if self.tokenize(candidate) > maximun_token:
                    break
                trimmed_meta.append(m)
            return "\n".join(trimmed_meta)

        # 컨텐츠는 순서 유지하며 가능한 만큼
        for c in content_lines:
            candidate = base + ("\n" if keep or meta_lines else "") + "\n".join(keep + [c])
            if self.tokenize(candidate) > maximun_token:
                break
            keep.append(c)

        return "\n".join(meta_lines + keep)