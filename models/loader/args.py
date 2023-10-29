import argparse
from configs.model_config import *


parser = argparse.ArgumentParser(prog='exoskeleton_armor', 
                                 description='personal knowledge base')

parser.add_argument('--no-remote-model', action='store_true', help='remote in the model on '
                                                                   'loader checkpoint, '
                                                                   'if your load local '
                                                                   'model to add the ` '
                                                                   '--no-remote-model` ')

parser.add_argument('--model-name', type=str, default=LLM_MODEL, help='Name of the model to load by default.')

# Accelerate/transformers
parser.add_argument('--load-in-8bit', action='store_true', default=LOAD_IN_8BIT,
                    help='Load the model with 8-bit precision.')
parser.add_argument('--bf16', action='store_true', default=BF16,
                    help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')

args = parser.parse_args([])
# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(args)
