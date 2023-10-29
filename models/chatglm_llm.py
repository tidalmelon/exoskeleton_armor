from typing import Any, Dict, List, Optional, Generator
from abc import ABC
import transformers

from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun

from models.base import (BaseAnswer, 
                         AnswerResultStream, 
                         AnswerResult, 
                         AnswerResultQueueSentinelTokenListenerQueue)

from models.loader import LoaderCheckPoint


class ChatGLMLLMChain(BaseAnswer, Chain, ABC):

    # 代表返回结果的token数量.
    max_token: int = 10000
    # 范围0-2, 默认1, 
    # 数值越小，模型倾向于高概率词汇. 文本更保守, 更加严谨， 更加严肃
    # 数值越大，模型倾向于低概率词汇. 文本更多样， 更活泼，甚至胡说八道
    temperature: float = 0.01
    # 相关度
    # 用于控制文本的随机性, 趋近于0，随机性越弱. 趋近于1，随机性越强.
    # P采样：
    top_p = 0.4
    # 候选词数量
    top_k = 10
    checkPoint: LoaderCheckPoint = None

    history_len: int = 10
    streaming_key: str = "streaming"
    history_key: str = "history"
    prompt_key: str = "prompt"
    output_key: str = "answer_result_stream"

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    #@abstractmethod from BaseAnswer
    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    #@abstractmethod from Chain
    @property
    def _chain_type(self) -> str:
        return "ChatGLMChain"

    #@abstractmethod from Chain
    @property
    def input_keys(self) -> List[str]:
        return [self.prompt_key]

    #@abstractmethod from Chain
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
            ) -> Dict[str, Generator]:
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        return {self.output_key: generator}

    #@abstractmethod from BaseAnswer: 真正调用模型产生答案
    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:

        history = inputs[self.history_key]
        streaming = inputs[self.streaming_key]
        prompt = inputs[self.prompt_key]

        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        # 本来是用来实现LLM.generate提前停止功能的。但这里return False, 不会提前停止
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)

        if streaming:

            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 0 else [], # 保留10个history
                    max_length = self.max_token,
                    temperature = self.temperature,
                    top_p = self.top_p,
                    top_k = self.top_k,
                    stopping_criteria = stopping_criteria_list
                    )):

                history[-1] = [prompt, stream_resp]
                # 消息实体
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}

                generate_with_callback(answer_result)
            self.checkPoint.clear_torch_cache()
        else:
            response, _ = self.checkPoint.model.chat(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stopping_criteria=stopping_criteria_list
                    )
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {'answer': response}

            generate_with_callback(answer_result)

