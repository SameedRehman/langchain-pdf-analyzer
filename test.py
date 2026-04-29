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
docs = load_any_file("/home/sameed/langchain_project/The history of monsters.txt")
# print(docs[0].page_content)
monster_text = docs[0].page_content



pdf_loader = PyPDFLoader("/home/sameed/Downloads/Atomic_Habits_100_Page_Mimic.pdf")
pdf_docs = load_any_file("/home/sameed/Downloads/Atomic_Habits_100_Page_Mimic.pdf")

print(f"Loaded {len(pdf_docs)} pages from the document")
pdf_text = pdf_docs[0].page_content + pdf_docs[1].page_content

context_text = monster_text +"\n\n----NEW DOCUMENT---\n\n" + pdf_text

# response = llm.invoke(prompt,"what is coding")

combined_question = """
1. What is the main point of the 'Atomic Habits' section?
2. Write a 5-line conclusion summarizing the 'History of Monsters' section.
"""

chain = prompt | llm  | output_parser

result = chain.invoke({
    "context":context_text,
    "question": combined_question
    })

print(result)