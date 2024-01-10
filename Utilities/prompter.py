from typing import Union
import json
import os.path as osp
import re


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "QASC-Full"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)

    def generate_prompt(
        self,
        question: str,
        answers: str,
        fact_1: Union[None, str] = None,
        fact_2: Union[None, str] = None,
        deduction: Union[None, str] = None,
        answer: Union[None, str] = None,
        label: Union[None, dict] = None,
        ) -> str:
    
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if deduction is None and all([fact_1, fact_2]):
            context = self.template["context_template"].format(question=question,
                                                           answers=answers,
                                                           fact_1=fact_1,
                                                           fact_2=fact_2)
            
        elif all([fact_1, fact_2, deduction]):
            context = self.template["context_template"].format(question=question,
                                                           answers=answers,
                                                           fact_1=fact_1,
                                                           fact_2=fact_2,
                                                           deduction=deduction)
            
        elif fact_1 is None and fact_2 is None and deduction is None:
            context = self.template["context_template"].format(question=question,
                                                           answers=answers)
            
            
        elif deduction is None and fact_2 is None:
            context = self.template["context_template"].format(question=question,
                                                               answers=answers,
                                                               fact_1=fact_1)
            
        elif deduction is None and fact_1 is None:
            context = self.template["context_template"].format(question=question,
                                                               answers=answers,
                                                               fact_2=fact_2)
        
            
            
        res = self.template["prompt"]
        res = f"{res}{context}"
        if label:
            label = self.template["steps_template"].format(**label)
            res = f"{res}{label}"
            
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        response_split = self.template["response_split"]
        output = output.split(response_split)[-1].strip()
        # try:
        #     output = output.split("\n\n")[0].strip()
        # except:
        #     output
        # result = output.split('\n')[:-1]
        # result = '\n'.join(result)
        pattern = re.compile('<.*?>')
        output = re.sub(pattern, '', output)
        return output
