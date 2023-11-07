import os

import datetime
from typing import List
from functools import lru_cache



from pypinyin import lazy_pinyin
from tqdm import tqdm

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.base import Chain
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter


from vectorstores import MyFAISS
from textsplitter.zh_title_enhance import zh_title_enhance
from textsplitter import ChineseTextSplitter
from configs.model_config import (VECTOR_SEARCH_TOP_K,
                                  VECTOR_SEARCH_SCORE_THRESHOLD,
                                  EMBEDDING_MODEL,
                                  EMBEDDING_DEVICE,
                                  CHUNK_SIZE,
                                  embedding_model_dict,
                                  CACHED_VS_NUM,
                                  PROMPT_TEMPLATE,
                                  SENTENCE_SIZE,
                                  STREAMING,
                                  ZH_TITLE_ENHANCE,
                                  KB_ROOT_PATH,
                                  logger,
                                  )
from utils import torch_gc






# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)

HuggingFaceEmbeddings.__hash__ = _embeddings_hash





# will keep CACHED_VS_NUM of vector store caches
# @lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path, embeddings):
    return MyFAISS.load_local(vs_path, embeddings)


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt

def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def load_file(filepath, sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE):

    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        # 如果你希望一段就是一个chunk, 那就将chunk_size, chunk_overlap都设置为0即可.
        # textsplitter = CharacterTextSplitter(separator='\n\n', chunk_size=0, chunk_overlap=0)
        docs = loader.load_and_split(textsplitter)
    #elif filepath.lower().endswith(".pdf"):
    #    # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
    #    from loader import UnstructuredPaddlePDFLoader
    #    loader = UnstructuredPaddlePDFLoader(filepath)
    #    textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
    #    docs = loader.load_and_split(textsplitter)
    #elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
    #    # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
    #    from loader import UnstructuredPaddleImageLoader
    #    loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
    #    textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
    #    docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    write_check_file(filepath, docs)
    return docs

def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    print('write_check_file', fp)
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()

class LocalDocQA:
    # query      查询内容
    # vs_path    知识库路径
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度

    llm_model_chain: Chain = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    # 是否启用上下文关联
    chunk_content_expand: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self, 
                 llm_model_chain: Chain = None,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm_model_chain = llm_model_chain
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            # 如果单个文件（文件夹）不存在
            if not os.path.exists(filepath):
                print(f"path {filepath} does not exist")
            # 如果单个文件
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath, sentence_size)
                    logger.info(f"{file} loaded success!")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} loaded failed")
            # 如果单个文件夹
            elif os.path.isdir(filepath):
                docs = []
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc='loading files'):
                    print('----', fullfilepath, file)
                    #try:
                    docs += load_file(fullfilepath, sentence_size)
                    logger.info(f"{file} loaded success!")
                    loaded_files.append(fullfilepath)
                    #except Exception as e:
                    #    logger.error(e)
                    #    failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("the following files were not loaded:")
                    for file in failed_files:
                        logger.info(f"{file}\n")
        # 如果是List, 则只能是文件列表，不能是文件夹列表
        else:

            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} loaded success!")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} loaded failed")
        # 以上是读取文件，文件夹，或者文件列表
        # 如果是文件夹会递归的读取所有文件， 如果参数是文件列表， 则不允许有文件夹

        if len(docs) > 0:
            logger.info("file has been loaded completely, in progress of generating vectorstore")
            if vs_path and os.path.isdir(vs_path) and 'index.faiss' in os.listdir(vs_path):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                torch_gc()
            else:
                if not vs_path:
                    vs_path = os.path.join(KB_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",
                                           "vector_store")
                vector_store = MyFAISS.from_documents(docs, self.embeddings) 
                torch_gc()

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            logger.info("all files load fail, please check the files")
            return None, loaded_files


    def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):

        print('get_knowledge_based_answer vs_path', vs_path)
        
        # 加载向量数据库
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_content_expand = self.chunk_content_expand
        vector_store.score_threshold = self.score_threshold


        print('vector_store:', vector_store)

        # 查最相似的top_k个chunks，即chunk的上一段，下一段。
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        print('related_docs_with_score', related_docs_with_score)
        torch_gc()
        if len(related_docs_with_score) > 0:
            # propmt 工程：将查询与chunks一起交给prompt
            prompt = generate_prompt(related_docs_with_score, query)
            print('search from vs success')
            print(prompt)
        else:
            # 否则就是原始提问
            prompt = query
            print('search from vs failed')
            print(prompt)

        answer_result_stream_result = self.llm_model_chain(
            {"prompt": prompt, "history": chat_history, "streaming": streaming})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history











































