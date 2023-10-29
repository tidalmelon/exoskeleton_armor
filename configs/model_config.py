import torch
import os







LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")





# LLM name
LLM_MODEL = 'chatglm2-6b-32k'
# load 8bit quantized model
LOAD_IN_8BIT = True
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False

# supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
# 如将 "chatglm-6b" 的 "local_model_path" 由 None 修改为 "User/Downloads/chatglm-6b"
# 此处请写绝对路径,且路径中必须包含repo-id的模型名称，因为FastChat是以模型名匹配的
llm_model_dict = {
    "chatglm2-6b-32k": {
        "name": "chatglm2-6b-32k",
        "pretrained_model_name": "THUDM/chatglm2-6b-32k",
        "local_model_path": '/root/.cache/huggingface/hub/THUDM/chatglm2-6b-32k',
        "provides": "ChatGLMLLMChain"
    },
}

