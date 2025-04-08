# full_app.py - Complete FastAPI Application

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, os, base64, uuid, io, time, random, re
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Helper Functions ---
def extract_elements(file_path):
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir="extracted_data"
    )
    categorized = {"Header": [], "Footer": [], "Title": [], "NarrativeText": [], "Text": [], "ListItem": [], "Table": [], "Image": []}
    for e in elements:
        t = str(type(e))
        for key in categorized:
            if key in t:
                categorized[key].append(str(e))
    return categorized

def summarize_texts(texts):
    prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text elements. \
        Give a concise summary of the table or text that is well optimized for retrieval.text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GEMINI_API_KEY)
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return chain.batch(texts, {"max_concurrency": 5})

def summarize_tables(tables):
    prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
        These summaries will be embedded and used to retrieve the raw table elements. \
        Give a concise summary of the table that is well optimized for retrieval. Table:{element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return chain.batch(tables, {"max_concurrency": 5})

def resize_and_encode_image(image_path, max_size=(256, 256), quality=70):
    try:
        img = Image.open(image_path).convert('L')
        img.thumbnail(max_size)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=quality)
        return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def image_summarize(base64_image, prompt):
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    max_retries, retry_delay, max_wait = 5, 2, 60
    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": base64_image}])
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "token" in str(e).lower() or "429" in str(e):
                time.sleep(retry_delay + random.random())
                retry_delay = min(retry_delay * 2, max_wait)
            else:
                raise e
    return "Image summary failed."

def generate_img_summaries(folder_path):
    img_b64, img_summaries = [], []
    prompt = "Summarize the image for retrieval in 2 sentences."
    for img_file in sorted(os.listdir(folder_path)):
        if img_file.endswith(".jpg"):
            path = os.path.join(folder_path, img_file)
            encoded = resize_and_encode_image(path)
            if encoded:
                img_b64.append(encoded)
                img_summaries.append(image_summarize(encoded, prompt))
    return img_b64, img_summaries

def build_multi_vector_retriever(text_summaries, texts, table_summaries, tables, img_summaries, images):
    id_key = "doc_id"
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    def add_docs(summaries, contents):
        doc_ids = [str(uuid.uuid4()) for _ in contents]
        summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, contents)))

    if text_summaries: add_docs(text_summaries, texts)
    if table_summaries: add_docs(table_summaries, tables)
    if img_summaries: add_docs(img_summaries, images)
    return retriever

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    file_path = Path("uploads") / file.filename
    Path("uploads").mkdir(exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    data = extract_elements(str(file_path))
    img_b64, img_summaries = generate_img_summaries("extracted_data")
    table_summaries = summarize_tables(data["Table"])
    text_summaries = summarize_texts(data["Text"])
    retriever = build_multi_vector_retriever(text_summaries, data["Text"], table_summaries, data["Table"], img_summaries, img_b64)

    return {"message": "PDF processed and retriever built."}


retriever_instance = None  
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GEMINI_API_KEY)

def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures)
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    b64_images, texts = [], []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    messages.append({"type": "text", "text": f"You are a helpful assistant.\nUser question: {data_dict['question']}\n\nText and tables:\n{formatted_texts}"})
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain(retriever):
    return (
        {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

@app.post("/query")
async def query_model(question: str):
    if not retriever_instance:
        return {"error": "Retriever not initialized. Upload a PDF first."}
    chain = multi_modal_rag_chain(retriever_instance)
    answer = chain.invoke(question)
    return {"response": answer}

# MODIFY PDF route to update retriever instance
def process_and_store(file_path):
    global retriever_instance
    data = extract_elements(str(file_path))
    img_b64, img_summaries = generate_img_summaries("extracted_data")
    table_summaries = summarize_tables(data["Table"])
    text_summaries = summarize_texts(data["Text"])
    retriever_instance = build_multi_vector_retriever(text_summaries, data["Text"], table_summaries, data["Table"], img_summaries, img_b64)

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    file_path = Path("uploads") / file.filename
    Path("uploads").mkdir(exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    process_and_store(file_path)
    return {"message": "PDF processed and retriever built."}
