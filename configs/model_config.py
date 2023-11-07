import os
import logging

import torch

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)





# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 5
# 匹配后单段上下文长度
CHUNK_SIZE = 250
# 传入LLM的历史记录长度
LLM_HISTORY_LEN = 3
# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，建议设置为500左右，经测试设置为小于500时，匹配结果更精准
# 500 这个阈值还是合理的。 大于500的往往相关度很低了. 设置为500主要起到过滤作用
# 相关度分数 > 500 的将被剔除. 因为这里采用的时欧式距离，越小相关度越大。
# MyFAISS: if i == -1 or 0 < self.score_threshold < scores[0][j]:
VECTOR_SEARCH_SCORE_THRESHOLD = 500
# Embedding model name
EMBEDDING_MODEL = "text2vec"
# Embedding running device
#EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EMBEDDING_DEVICE = "cpu"
# 缓存知识库数量,如果是ChatGLM2,ChatGLM2-int4,ChatGLM2-int8模型若检索效果不好可以调成’10’
CACHED_VS_NUM = 1
# 文本分句长度
SENTENCE_SIZE = 100
# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False

# LLM streaming reponse
STREAMING = True

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")





# LLM name
LLM_MODEL = 'chatglm2-6b-32k'
# load 8bit quantized model
LOAD_IN_8BIT = True
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""


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

# 在以下字典中修改属性值，以指定本地embedding模型存储位置
# 如将 "text2vec": "GanymedeNil/text2vec-large-chinese" 修改为 "text2vec": "User/Downloads/text2vec-large-chinese"
# 此处请写绝对路径
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    #"text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec": "/root/.cache/huggingface/hub/dataroot/models/GanymedeNil/text2vec-large-chinese",
    "text2vec-base-multilingual": "shibing624/text2vec-base-multilingual",
    "text2vec-base-chinese-sentence": "shibing624/text2vec-base-chinese-sentence",
    "text2vec-base-chinese-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
}

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
""")

