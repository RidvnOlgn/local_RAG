###  IMPORTS  ###

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

### LOAD KNOWLEDGEBASE ###
#  

def load_knowledgeBase(doc_path): #your pdf file path

    #load pdf file
    loader3 = PyPDFLoader(doc_path) 
    knowledgeBase = loader3.load_and_split()
    print('PDF File Read')

    return knowledgeBase


### CHUNKING ###
#

def chunking(knowledgeBase):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts_1024 = text_splitter.split_documents(knowledgeBase)
    #print(len(texts_1024))
    print('Chunking Done!')
    return texts_1024


### Import LLM Model ###
#
def llm_model(model_path):
    # Callbacks
    callbacks = [StreamingStdOutCallbackHandler()]

    llm= GPT4All(model=model_path, callbacks=callbacks, verbose=True,n_threads=16, temp=0.5)

    return llm

### Prompt Template ###
#
def prompt_Temp():

    prompt_template='''
    You can access the context between BEGININPUT and ENDINPUT tags for the following task. Answer the following question in English only, based on the given context. If these do not contain an answer, say that no answer is possible based on the information given!
    USER: 
    BEGININPUT{context}ENDINPUT
    BEGINING {question} END 
    ASSISTANT:
    '''
    PROMPT=PromptTemplate(
        template=prompt_template,input_variables=["context","question"]
    )

    return PROMPT


def qa_Rag_chain_fun(PROMPT,llm,db):
    chain_type_kwargs={"prompt":PROMPT}
    qa_RAG_chain =RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=False,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_RAG_chain

def main():

    #Your pdf file path
    path = "BodyPartRecognition.pdf"

    knowledgeBase = load_knowledgeBase(path)

    texts = chunking(knowledgeBase)

    # embedings model name
    #model_name = "google-bert/bert-base-uncased"
    #model_name = "sentence-transformers/all-roberta-large-v1"
    model_name = "sentence-transformers/msmarco-bert-base-dot-v5"

    #DB name
    db_name = "chroma_db_wizard"
    
    #create emmbedings model
    model_ = HuggingFaceEmbeddings(model_name= model_name)

    #create chroma db
    db_path = os.path.join(db_name)
    if os.path.exists(db_path):
        db=Chroma(persist_directory="chroma_db_wizard", embedding_function=model_)
    else:
        db = Chroma.from_documents(texts, model_, persist_directory="chroma_db_wizard")

    #local model path
    model_path = "C:/Users/Ridvan/AppData/Local/nomic.ai/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf"

    llm = llm_model(model_path)

    #promp template
    prompt = prompt_Temp()

    qa_Rag = qa_Rag_chain_fun(prompt,llm,db)

    print("Hello! I am an assistant developed with the RAG method. You can ask questions related to your documents. You can type 'exit' or 'q' to exit.")
    while True:
        user_input = input("Question: ")
        if user_input.lower() == 'exit' or user_input.lower() == 'q':
            print("Assistant is closing, By.")
            break
        print("Answer: ", end="") 
        qa_Rag.invoke(user_input)["result"]
        print()
    

if __name__ == "__main__":
    main()