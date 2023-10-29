import gc
import re
from typing import Optional, List, Dict, Tuple, Union

from pathlib import Path
from configs.model_config import LLM_DEVICE

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LoaderCheckPoint:
    """
    加载自定义 model CheckPoint
    """

    # remote in the model on loader checkpoint
    no_remote_model: bool = False
    # checkpoint
    pretrained_model_name: str = None
    local_model_path: str = None
    tokenizer: object = None
    model_config: object = None
    params: object = None
    llm_device = LLM_DEVICE # model_config: LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # mapping: layers->gpu_num, this field is manually defined, not auto defined
    device_map: Optional[Dict[str, int]] = None


    def __init__(self, params: Dict = None):

        self.params = params or {}

        self.tokenizer = None

        self.model_name = params['model_name']
        # un-indentitied
        self.no_remote_model = params.get('no_remote_model', False)
        self.load_in_8bit = params.get('load_in_8bit', False)


    def _get_checkpoint(self):
        if self.local_model_path:
            return self.local_model_path
        else:
            return self.pretrained_model_name


    def _load_model_config(self):

        # 

        checkpoint = self._get_checkpoint()

        self.model_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)

        return self.model_config


    def _load_model(self):
        """
        加载自定义位置的model
        : return: 
        """


        checkpoint = self._get_checkpoint()


        if 'chatglm' in self.model_name.lower() or 'chatyuan' in self.model_name.lower():
            LoaderClass = AutoModel
        

        if self.load_in_8bit:

            from accelerate import init_empty_weights
            from accelerate.utils import get_balanced_memory, infer_auto_device_map
            from transformers import BitsAndBytesConfig

            params = {'low_cpu_mem_usage': True}

            if not self.llm_device.lower().startswith('cuda'):
                raise SystemError("8bit mode needs CUDA supporting, or you can adopt quantization  model")
            else:
                params['device_map'] = 'auto'
                params['trust_remote_code'] = True

                """
                # 量化配置
                quantization_config (`Dict`, *optional*):
                    A dictionary of configuration parameters for the `bitsandbytes` library and loading the model using
                    advanced features such as offloading in fp32 on CPU or on disk.

                load_in_8bit (`bool`, *optional*, defaults to `False`):
                    This flag is used to enable 8-bit quantization with LLM.int8().


                llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
                    This flag is used for advanced use cases and users that are aware of this feature. If you want to split
                    your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
                    this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
                    operations will not be run on CPU.
                """

                # 相当于开启量化8位模式
                params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True,
                                                                   llm_int8_enable_fp32_cpu_offload=False)

            # 任何模型，无论大小，都可以在此方法的上下文 (context) 内进行初始化，而无需为模型权重分配任何内存
            with init_empty_weights():
                model = LoaderClass.from_config(self.model_config, trust_remote_code=True)

            # 系住权重
            model.tie_weights()

            # 这里是实际的量化加载
            try:
                model = LoaderClass.from_pretrained(checkpoint, **params)
            except ImportError as exc:
                raise ValueError(
                        "if you enable 8bits quantization version and the  project is failed to start" \
                        "please refer to this url and chose the fitable cuda version, " \
                        "https://github.com/TimDettmers/bitsandbytes/issues/156"
                ) from exc

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

        return model, tokenizer


    def reload_model(self):

        self.model_config = self._load_model_config()

        self.model, self.tokenizer = self._load_model()

        self.model = self.model.eval()




    def clear_torch_cache(self):
        gc.collect()
        
        # 配置文件中指定的CPU cuda cuda:2
        if self.llm_device.lower() != "cpu":

            if torch.backends.mps.is_built():
                try:
                    from torch.mps import empty_cache
                    empty_cache()
                except Exception as e:
                    print(e)
                    print(
                        "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
            elif torch.backends.cuda.is_built():
                device_id = "0" if torch.cuda.is_available() and (":" not in self.llm_device) else None
                # cuda-> cuda:0 cuda:1 不变
                CUDA_DEVICE = f"{self.llm_device}:{device_id}" if device_id else self.llm_device
                with torch.cuda.device(CUDA_DEVICE):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            else:
                print("cannot detect cuda or mps")




















