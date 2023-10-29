
import sys
from typing import Any


from models.loader import LoaderCheckPoint
from configs.model_config import llm_model_dict


loaderCheckPoint: LoaderCheckPoint = None


def loaderLLM(llm_model: str = None, no_remote_model: bool = False) -> Any:
    """
    init llm_model_ins LLM
    :param llm_model: model_name
    :param no_remote_model: remote in the model on loader checkpoint, if your load local model to add the 
                            `--no-remote-model`
    :param use_ptuning_v2: use p-tuning-v2 prefixEncoder
    :return 
    """

    # pre_model_name 从哪来的

    pre_model_name = loaderCheckPoint.model_name
    llm_model_info = llm_model_dict[pre_model_name]



    if no_remote_model:
        loaderCheckPoint.no_remote_model = no_remote_model

    if llm_model:
        llm_model_info = llm_model_dict[llm_model]

    loaderCheckPoint.model_name = llm_model_info['name']
    loaderCheckPoint.pretrained_model_name = llm_model_info['pretrained_model_name']
    loaderCheckPoint.local_model_path = llm_model_info['local_model_path']

    loaderCheckPoint.reload_model()


    # 反射加载类
    # getattr(sys.modules[name], func_name)
    # 找到当前文件下名称为func_name的对象（类对象或者函数对象）
    # from models, 多次引入过models module
    provides_class = getattr(sys.modules['models'], llm_model_info['provides'])
    print('provides_class is ', provides_class)

    modelInsLLM = provides_class(checkPoint=loaderCheckPoint)

    #  llm_model_info['provides']
    #  provides:  ChatGLMLLMChain # 这个是任务链,继承自langchain
    #  sys.modules['models']
    #  'models' : '/home/10170464/jupyter-notebook/Wangquanjun/langchain-ChatGLM/models/__init__.py'

    return modelInsLLM








































