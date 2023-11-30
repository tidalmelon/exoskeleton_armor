import sys
sys.path.append("/home/wangquanjun/project/websites/starchain/llm_server/chatglm")
from chatglm_extension import ChatGLM

sys.path.append("/home/wangquanjun/project/websites/starchain/llm_server/utils")
from utils import remove_blank_keep_newline


from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.schema.document import Document

import os
import time
import datetime
import json

import streamlit as st
from streamlit_chatbox import ChatBox, Markdown


"""
http://www.360doc.com/content/23/0511/14/75507148_1080268383.shtml
* 的作用
"""

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:

    title = "简历标准化"
    st.set_page_config(page_title=title, layout="wide")  # header title


    path = st.text_input("简历路径", max_chars=None, key=None, type="default") # default
    path = path.strip()


    saved = st.button("开始分析")

    #path = '/home/wangquanjun/jupyter/app_test/01.ResumeFilter/data/三人行简历-佳晟-熔铸-王仁杰.doc'
    

    if saved:

        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()


        text = docs[0].page_content[:500]
        page_content = remove_blank_keep_newline(text)
        doc_related = Document(page_content=page_content, metadata={'source': path})


        options = ["姓名", "性别", "身份证", "民族", "出生日期", "地址", "学历", "专业", "邮箱", "联系方式", "是否在职"]
        option = ','.join(options)
        question = f"这个应聘者的{option}分别是什么，请以json形式返回结果，最多不超过250字，如果查找不到，则返回'未提供'"


        st.write(question)


        llm = ChatGLM()
        llm.endpoint_url = 'http://0.0.0.0:4002'
        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain({'input_documents': [doc_related], 'question': question})
        result = response['output_text']
        try:
            dic = json.loads(reponse['output_text'].strip('"').strip().replace("\\n", "").replace("\\", "").replace("，", ','))
            result = json.dumps(dic, indent=4)
        except:
            pass

        st.write(result)

        text = docs[0].page_content[500:]
        page_content = remove_blank_keep_newline(text)
        doc_related = Document(page_content=page_content, metadata={'source': path})


        options = ["是否在职", "当前职务", "技术能力", "经验程度", "管理能力"]
        option = ','.join(options)
        question = f"这个应聘者的{option}分别是什么，请以json形式返回结果，最多不超过250字，如果查找不到，则返回'未提供'"

        st.write(question)

        llm = ChatGLM()
        llm.endpoint_url = 'http://0.0.0.0:4002'
        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain({'input_documents': [doc_related], 'question': question})
        result = response['output_text']
        try:
            dic = json.loads(reponse['output_text'].strip('"').strip().replace("\\n", "").replace("\\", "").replace("，", ','))
            result = json.dumps(dic, indent=4)
        except:
            pass
        st.write(result)

        question = "能抽取应聘者的工作履历么， 请以json形式返回结果, 最多不超过500字，如果查找不到，则返回'未提供'"
        st.write(question)
        llm = ChatGLM()
        llm.endpoint_url = 'http://0.0.0.0:4002'
        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain({'input_documents': [doc_related], 'question': question})
        result = response['output_text']
        try:
            dic = json.loads(reponse['output_text'].strip('"').strip().replace("\\n", "").replace("\\", "").replace("，", ',').replace(" ", ""))
            result = json.dumps(dic, indent=4)
        except:
            pass

        st.write(result)


else:
    st.warning('Login First')



