import os
import json
import uvicorn
import logging
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

import zhipuai

import sys
sys.path.append("/home/wangquanjun/project/websites/starchain/llm_server/utils")
from log_util import get_logger


logger = get_logger('./logs/chatglm_online')

MAX_HISTORY = 3


class ChatGLMOnline():
    """
    暂不支持多轮对话记忆功能, history即包含user也需要记录assistant角色
    """
    def __init__(self) -> None:
        logger.info("Start initialize model...")
        zhipuai.api_key = "4738bb4eb23520f66f1f5ca6a68659fd.sMNQwf2toLjFp1O1"
        logger.info("Model initialization finished.")

    def answer(self, prompt, history=[], max_length=2048, temperature=0.95, top_p=0.7):

        logger.info(f"max_length={max_length}, temperature={temperature}, top_p={top_p}")

        response = zhipuai.model_api.invoke(
                            model="chatglm_turbo",
                            prompt=[{"role": "user", "content": prompt}],
                            temperature=temperature, 
                            top_p=top_p,
                            #incremental=False,
                            #return_type='json_string'
                            )

        history = [list(h) for h in history]
        return response['data']['choices'][0]['content'], history

    def stream(self, prompt, history=[], max_length=2048, temperature=0.95, top_p=0.7):

        logger.info(f"max_length={max_length}, temperature={temperature}, top_p={top_p}")

        if prompt is None:
            yield {"prompt": "", "response": "", "history": [], "finished": True}

        response = zhipuai.model_api.sse_invoke(
                            model="chatglm_turbo",
                            prompt=[{"role": "user", "content": prompt}],
                            temperature=temperature, 
                            top_p=top_p,
                            #incremental=False,
                            #return_type='json_string'
                            )

        buffer = ""
        for event in response.events():
            if event.event == "add":
                buffer += event.data
                yield {"delta": event.data, "response": buffer, "finished": False}
            elif event.event == "error" or event.event == "interrupted":
                buffer += event.data
                yield {"delta": event.data, "response": buffer, "finished": False}
            elif event.event == "finish":
                yield {"prompt": prompt, "delta": "[EOS]", "response": buffer, "history": history, "finished": True}
            else:
                # 理论上不会被执行到
                logger.info("impossiable branch...")



def start_server(http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = ChatGLMOnline()

    app = FastAPI()
    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"]
                       )

    @app.get("/")
    def index():
        return {'message': 'started', 'success': True}

    @app.post("/chat")
    async def answer_question(arg_dict: dict):
        result = {"prompt": "", "response": "", "success": False}
        try:
            text = arg_dict.pop("prompt")
            ori_history = arg_dict.pop("history")
            logger.info("Prompt - {}".format(text))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))

            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            response, history = bot.answer(text, history, **arg_dict)
            logger.info("Answer - {}".format(response))
            ori_history.append((text, response))
            result = {"prompt": text, "response": response,
                      "history": ori_history, "success": True}
        except Exception as e:
            logger.error(f"error: {e}")
        return result

    @app.post("/stream")
    def answer_question_stream(arg_dict: dict):
        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')

        try:
            text = arg_dict.pop("prompt")
            ori_history = arg_dict.pop("history")
            logger.info("Prompt - {}".format(text))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            return EventSourceResponse(decorate(bot.stream(text, history, **arg_dict)))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(decorate(bot.stream(None, None)))


    @app.get("/free_gc")
    def free_gpu_cache():
        pass

    logger.info("starting server...")
    uvicorn.run(app=app, host=http_address, port=port, workers=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for ChatGLM2-6B')
    parser.add_argument('--device', '-d', help='device，-1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=4002)
    args = parser.parse_args()
    start_server(args.host, int(args.port), args.device)















