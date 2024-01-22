import nltk

from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
from configs.model_config import (NLTK_DATA_PATH,
                                  LLM_HISTORY_LEN,
                                  STREAMING,
                                  )
from chains.local_doc_qa import LocalDocQA




nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def test_llm():
    #history = []
    #last_print_len = 0
    #for response, history in llm_model_ins.model.stream_chat(llm_model_ins.tokenizer, query, history):

    #    print(response[last_print_len:], end="", flush=True)
    #    last_print_len = len(response)

    pass


def test_llmchain():

    llm_model_chain_ins = shared.loaderLLM()

    query = "AIGC对教育有什么影响"

    context = """
    首先，AIGC的出现使得学生的优势不再仅限于特定的知识性技能。在过去，学生需要花费大量时间和精力来记忆和掌握各种知识。然而，随着AIGC技术的不断发展，很多知识性的技能可能被AI所替代。因此，未来教育的首要目标应该是培养具有独立思考、正确价值判断能力和创新能力的人。

其次，AIGC的出现将打破传统千篇一律的教育模式，让学生的个性化得到充分发展。每个学生都有自己独特的兴趣和能力，但在传统的教育模式下，学生往往只能按照统一的课程和要求进行学习，无法充分发挥自己的潜力。而AIGC可以通过智能推荐等技术，引导学生主动探索和研究自己感兴趣的问题，从而培养学生的创新能力和独立思考能力。

同时，未来的教育一定是借助AI工具自我学习的时代。学生可以借助AIGC工具，根据自己的基础、需求和兴趣定时学习内容，从而获得更好的学习体验和学习效果，充分释放潜力。这种个性化的学习方式可以让学生更加专注于自己的兴趣和特长，提高学习效果和兴趣。

然而，如何让孩子的潜力得到更大程度的发挥，成为每个家长研究的课题。家长需要了解孩子的兴趣和需求，选择适合孩子的AIGC工具，并引导孩子正确地使用这些工具。同时，家长还需要关注孩子的学习进展和反馈，及时调整孩子的学习计划和方法。

最后，教育者和政策制定者也需要认真思考如何利用AIGC工具来改进教育方式和提高教育质量。他们需要关注学生的学习需求和兴趣，提供更加个性化和灵活的教育方式。同时，他们还需要关注AIGC工具的质量和准确性，确保学生获得正确和可靠的学习资源。
    """

    # 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
    PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

    prompt = PROMPT_TEMPLATE.replace("{question}", query).replace("{context}", context)

    #from langchain.prompts import PromptTemplate
    #prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    #prompt = prompt.format(question=query, context=context)


    history = []
    answer_result_stream_result = llm_model_chain_ins(
            {"prompt": prompt, "history": history, "streaming": True})


    last_print_len = 0
    for answer_result in answer_result_stream_result['answer_result_stream']:
        resp = answer_result.llm_output["answer"]
        print(resp[last_print_len:], end="", flush=True)
        last_print_len = len(resp)

    print()
    print('load success')

def test_local_kg_qa():
    llm_model_chain_ins = shared.loaderLLM()
    llm_model_chain_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model_chain=llm_model_chain_ins)

    vs_path = None
    filepath = './input_data/'

    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filepath=filepath)
    if vs_path:
        print(f'has created a new vs file: {vs_path}')

    history = []
    while True:
        query = input("Input your question 请输入问题：")
        #query = "三国演义的故事概要"
        #query = "AIGC对教育有什么影响"
        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=STREAMING):
            if STREAMING:
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                print(resp["result"])
        print()
        print()
        #if REPLY_WITH_SOURCE:
        #    source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
        #                   # f"""相关度：{doc.metadata['score']}\n\n"""
        #                   for inum, doc in
        #                   enumerate(resp["source_documents"])]
        #    print("\n\n" + "\n\n".join(source_text))



if __name__ == '__main__':
    args = None
    args = parser.parse_args()
    args_dict = vars(args)

    print(args_dict)

    #shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    #test_llmchain()
    #test_local_kg_qa()














