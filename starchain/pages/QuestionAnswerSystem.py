import sys

sys.path.append("/home/wangquanjun/project/websites/starchain/llm_server/chatglm")

from chatglm_extension import ChatGLM



import streamlit as st
from streamlit_chatbox import ChatBox, Markdown


if 'authentication_status' in st.session_state and st.session_state['authentication_status']:

    title = "大语言模型知识库"
    st.set_page_config(page_title=title, layout="wide")  # header title

    #st.title(title) # H1 title

    chat_box = ChatBox()
    
    # 官网给出根据column可以放置到右边. 可以参考chatchat
    with st.sidebar:
        st.subheader("Start to chat using streamlit")
        streaming = st.checkbox("streaming", True)
        in_expander = st.checkbox("show messages in expander", True)
        show_history = st.checkbox("show history", False)
    
    
    
    chat_box.init_session()
    chat_box.output_messages()
    
    if query := st.chat_input('input your question here'):
        chat_box.user_say(query)
    
        # 未使用呢
        history = []

        llm = ChatGLM()
    
        elements = chat_box.ai_say(
                [
                    Markdown("", 
                             expanded=True, 
                             #in_expander=in_expander, # 是否显示转圈
                             #title="answer"
                             ),
                    #Markdown("", in_expander=in_expander, title="references"),
                ]
        )

        text = ""
        for chunk in llm.stream(query):
            text += chunk
            chat_box.update_msg(text, element_index=0, streaming=True)
    
        # update the element without focus
        chat_box.update_msg(text, element_index=0, streaming=False, state="complete")

else:
    st.warning('Login First')



