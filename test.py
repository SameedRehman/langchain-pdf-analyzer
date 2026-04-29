from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate.from_template("Answer the Following Questions based only on the provided text: \n\n Context:{context} \n\n Question:{question}")

output_parser = StrOutputParser()

def load_any_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader=TextLoader(file_path)
    else:
        return None
    return loader.load()

loader = TextLoader("The history of monsters.txt")
docs = loader.load()
print(docs[0].page_content)

pdf_loader = PyPDFLoader("/home/sameed/Downloads/Atomic_Habits_100_Page_Mimic.pdf")
pdf_docs = pdf_loader.load()

print(f"Loaded {len(pdf_docs)} pages from the document")

context_text = pdf_docs[0].page_content + pdf_docs[1].page_content

# response = llm.invoke(prompt,"what is coding")

chain = prompt | llm  | output_parser

result = chain.invoke({
    "context":context_text,
    "question": "What is the main point of the document"})

print(result)