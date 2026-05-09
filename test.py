import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone ,ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(dotenv_path=".env")
pinecone_api_key = os.getenv("PINECONE_API_KEY") # or "PINECONE_API_KEY"

if pinecone_api_key:
    pinecone_api_key=pinecone_api_key.strip().strip('"').strip("'")

print(f"DEBUG: Pinecone key found: {pinecone_api_key is not None}")

pc = Pinecone(api_key=pinecone_api_key)


llm = ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens = 1000,
    )

prompt = PromptTemplate.from_template("Answer the Following Questions based only on the provided text: \n\n Context:{context} \n\n Question:{question}")

output_parser = StrOutputParser()

# Helper function to load files safely
def load_any_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        return None
    return loader.load()

pdf_file_path = ("Atomic_Habits_100_Page_Mimic.pdf")
pdf_doc = load_any_file(pdf_file_path)
# print(f"Loaded {len(pdf_doc)} pages from the document")

pdf_txt = ""

for doc in pdf_doc:
    pdf_txt += doc.page_content + "\n"

context_txt = pdf_txt

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

all_spplitter = txt_splitter.split_documents(pdf_doc)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


index_name = "first-key"

if index_name not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating new index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'  # <--- Change this from 'ap-southeast-1' to 'us-east-1'
        )
    )
    print("Index created successfully!")


vectorstore = PineconeVectorStore.from_documents(
    documents=all_spplitter,
    embedding=embeddings,
    index_name=index_name
)


query = input("Ask your question: ")
# query = "What does the PDF say about habbit?"

docs = vectorstore.similarity_search(query, k=3)

retriever =  vectorstore.as_retriever()

relevant_context_text ="\n\n".join([doc.page_content for doc in docs])

relevant_context_text = retriever.invoke(query) 




chain = prompt | llm  | output_parser

result = chain.invoke({
    "context":relevant_context_text,
    "question": query
    })

print("\n------ANSWER-------\n")
print(result)