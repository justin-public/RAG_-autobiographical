import faiss
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from hashlib import sha256

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from io import BytesIO

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

def text_field(label, columns=None, **input_params):
    c1, c2 = st.columns(columns or [1, 5])
    c1.markdown("##")
    c1.markdown(label)
    input_params.setdefault("key", label)
    return c2.text_input("", **input_params)

def fileload_field(label, columns=None, **input_params):
    c1, c2 = st.columns(columns or [1, 5])
    c1.markdown("##")
    c1.markdown("##")
    c1.markdown(label)
    input_params.setdefault("key", label)
    return c2.file_uploader("", **input_params)

def load_and_split_documents(loaders):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    all_splits = []
    
    for loader in loaders:
        pages = loader.load_and_split()
        splits = text_splitter.split_documents(pages)
        all_splits.extend(splits)
    return all_splits

def main():
    llm = ChatOllama(
        model="ggml-model-Q5_K_M/Modelfile:latest",
        callback_manager=CallbackManager([]),  # Empty list to disable streaming output
    )
    
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    
    st.markdown("자서전 생성")
    
    uploaded_file = fileload_field("기본 약력")
    qq_pdf_file = fileload_field("100문 100답")
    etc_file_file = fileload_field("기타")

    if qq_pdf_file is not None:
        qq_pdf_file_name = qq_pdf_file.name
        
    if etc_file_file is not None:
        etc_file_file_name = etc_file_file.name
        
    c1 ,c2, c3 = st.columns([1,3,4])
    with c1:
        with stylable_container(key="type", css_styles=["""{left: 60px;}""",]):
            st.markdown("형식")
    with c2:
        with stylable_container(key="checkbt", css_styles=["""{left: 100px;}""",]):
            st.checkbox("편년체")
    with c3:
        with stylable_container(key="checkbt1", css_styles=["""{left: 100px;}""",]):
            st.checkbox("주제별")

    contextualize_q_system_prompt = st.text_area("", key='text_value')
    qa_system_prompt = st.text_area("", key='text_value1')

    product_pdf = st.button("자서전 생성")

    if product_pdf:
        loaders = [
            PyPDFLoader(f'C:/work/langchain_llama#5/{qq_pdf_file_name}'),
            TextLoader(f'C:/work/langchain_llama#5/{etc_file_file_name}', encoding='UTF8')
        ]
        all_splits = load_and_split_documents(loaders)
        index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector = FAISS.from_documents(documents=all_splits, embedding=embedding_model)
        retriever = vector.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("system", "{context}"),  # context를 system 메시지로 추가
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        GHistorys = {}
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in GHistorys:
                GHistorys[session_id] = ChatMessageHistory()
            return GHistorys[session_id]
        
        def get_answer_from_session_result(result):
            return result.get('answer', '')

        answers = []

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # PDF 파일 생성
        pdfmetrics.registerFont(TTFont('NotoSans', 'NotoSansKR-Regular.ttf'))
        
        def create_pdf(title, prologue, chapter_title, chapter_text, chapter2_title, chapter2_text, chapter3_title, chapter3_text, chapter4_title, chapter4_text, file_path):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch, bottomMargin=inch)
            styles = getSampleStyleSheet()
            pdfmetrics.registerFont(TTFont('NotoSans', 'NotoSansKR-Regular.ttf'))

            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontName='NotoSans',
                fontSize=18,
                spaceAfter=12
            )

            prologue_style = ParagraphStyle(
                'Prologue',
                parent=styles['Normal'],
                fontName='NotoSans',
                fontSize=12,
                leading=18,
                alignment=TA_JUSTIFY  
            )

            chapter_title_style = ParagraphStyle(
                'ChapterTitle',
                parent=styles['Heading1'],
                fontName='NotoSans',
                fontSize=14,
                spaceBefore=12,
                spaceAfter=6
            )

            chapter_text_style = ParagraphStyle(
                'ChapterText',
                parent=styles['Normal'],
                fontName='NotoSans',
                fontSize=12,
                leading=18,
                alignment=TA_JUSTIFY  
            )

            # Prepare content
            content = [
                Paragraph(title, title_style),
                Spacer(1, 12),
                Paragraph("프롤로그", chapter_title_style),                     
                Spacer(1, 12),
                Paragraph(prologue, prologue_style),
                Spacer(1, 12),
                Paragraph(chapter_title, chapter_title_style),
                Spacer(1, 12),
                Paragraph(chapter_text, chapter_text_style),
                Spacer(1, 12),
                Paragraph(chapter2_title, chapter_title_style),
                Spacer(1, 12),
                Paragraph(chapter2_text, chapter_text_style),
                Spacer(1, 12),
                Paragraph(chapter3_title, chapter_title_style),
                Spacer(1, 12),
                Paragraph(chapter3_text, chapter_text_style),
                Spacer(1, 12),
                Paragraph(chapter4_title, chapter_title_style),
                Spacer(1, 12),
                Paragraph(chapter4_text, chapter_text_style),
            ]
            doc.build(content)
            buffer.seek(0)

            with open(file_path, 'wb') as f:
                f.write(buffer.read())

        def invoke_and_print(input_text, session_id, title,context=""):
            result = conversational_rag_chain.invoke(
                {"input": input_text, "context": context},  # context를 올바르게 전달
                config={"configurable": {"session_id": session_id}}
            )
            answer = get_answer_from_session_result(result)
            answers.append((title, answer))
            return answer
        
        
        
        prologue = invoke_and_print("100문 100답 내용을 기반으로 자서전에 프롤로그를 작성해주세요", "user_session_1", "")
        print(prologue)
        chapter1 = invoke_and_print("100문 100답 내용을 기반으로 자서전에 유년시절의 기억을 작성해주세요", "user_session_2", "")
        print(chapter1)
        chapter2 = invoke_and_print("100문 100답 내용을 기반으로 자서전에 청소년 시절의 도전과 성찰을 작성해주세요", "user_session_3", "")
        print(chapter2)
        chapter3 = invoke_and_print("100문 100답 내용을 기반으로 자서전에 첫사랑과 대학시절을 작성해주세요", "user_session_4", "")
        print(chapter3)
        chapter4 = invoke_and_print("100문 100답 내용을 기반으로 자서전에 결혼과 가정의 시작을 작성해주세요", "user_session_5", "")
        print(chapter4)

        create_pdf("사랑과도전으로가득한나의인생", prologue, "1장 유년시절의 기억들", chapter1, "2장 청소년 시절의 도전과 성찰", chapter2, "3장 첫사랑과 대학시절", chapter3, "4장 결혼과 가정의 시작", chapter4, "answers.pdf")
        
    
    
    
    st.markdown("자서전 요약본 생성")
    original_file_load = fileload_field("저서전 원본")

    col1, col2, col3, col4, col5 = st.columns([0.13, 0.2, 0.5, 0.5, 0.5])
    with col1:
        col1.markdown("")
    with col2:
        col2.markdown("#")
        col2.markdown("단어수")
    with col3:
        col3.text_input("", key="input_3")
    with col4:
        col4.markdown("#")
        col4.markdown("~")    
    with col5:
        col5.text_input("", key="input_4")

    st.button("자서전 요약본 생성")

if __name__ == "__main__":
    main()
