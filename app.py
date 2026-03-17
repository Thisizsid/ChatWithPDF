import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="centered")
st.title("📄 PDF Chat with OpenAI + Pinecone")
st.caption("Upload PDFs → Ask questions. Knowledge stored in Pinecone index 'chatwithpdf'")

# ── Check required environment variables ───────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found in environment / Streamlit secrets")
    st.stop()

if not pinecone_api_key:
    st.error("❌ PINECONE_API_KEY not found in environment / Streamlit secrets")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


embeddings = get_embeddings()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=openai_api_key,
    streaming=True,
)

# ── Session state ────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}

session_id = st.text_input("Session ID (for conversation memory)", value="default_session")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages.clear()
        if session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
        st.rerun()
    if st.session_state.vectorstore:
        st.caption("📊 Using Pinecone index: chatwithpdf-1536")

# ── PDF Upload ───────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True,
    help="Files will be added to your Pinecone index 'chatwithpdf-1536'",
)

INDEX_NAME = "chatwithpdf-1536"

if uploaded_files:
    with st.status("Processing PDFs and upserting to Pinecone...", expanded=True) as status:
        new_documents = []

        for file in uploaded_files:
            temp_path = f"./temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                new_documents.extend(docs)
                status.write(f"✅ Loaded: {file.name} ({len(docs)} pages)")
            except Exception as e:
                status.error(f"Failed {file.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if new_documents:
            status.write("Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            splits = text_splitter.split_documents(new_documents)

            status.write("Upserting to Pinecone index...")
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = PineconeVectorStore.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    index_name=INDEX_NAME,
                )
            else:
                st.session_state.vectorstore.add_documents(splits)

            status.update(
                label=f"✅ Upserted {len(splits)} chunks to Pinecone!",
                state="complete",
            )
        else:
            status.update(label="No new documents to process", state="complete")

# ── Connect to existing Pinecone index ──────────────────────────────────
if st.session_state.vectorstore is None:
    with st.spinner("Connecting to existing Pinecone index..."):
        try:
            st.session_state.vectorstore = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings,
            )
            st.success("Connected to Pinecone index", icon="✅")
        except Exception as e:
            st.error(f"Could not connect to Pinecone index: {str(e)}")
            st.stop()

# ── Build RAG chain ──────────────────────────────────────────────────────
# NOTE: MessagesPlaceholder(variable_name="chat_history") is mandatory here.
# Using a plain ("role", "{chat_history}") tuple causes the variable to resolve
# to None under langchain-core >= 0.3 + pydantic v2, crashing
# create_stuff_documents_chain with a NoneType TypeError.

@st.cache_resource(show_spinner="Building RAG chain...")
def build_rag_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 6})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "Given a chat history and the latest user question, "
                "formulate a standalone question that can be understood "
                "without the chat history. Do NOT answer it."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a helpful assistant answering questions based ONLY on the provided context.\n"
        "Summarize and explain the answer based on the context provided.\n"
        "If the context is partial, answer as best you can, but indicate if something is missing.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


conversational_rag_chain = None

if st.session_state.vectorstore is not None:
    try:
        rag_chain = build_rag_chain(st.session_state.vectorstore)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    except Exception as e:
        st.error(f"Failed to build RAG chain: {str(e)}")
        st.stop()

# ── Chat interface ───────────────────────────────────────────────────────
if conversational_rag_chain is None:
    st.info("👆 Upload at least one PDF to start chatting (or wait while connecting to Pinecone).")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about the uploaded PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            def stream_response():
                full_response = ""
                context_docs = []

                for chunk in conversational_rag_chain.stream(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}},
                ):
                    if "context" in chunk and chunk["context"]:
                        context_docs = chunk["context"]
                    if "answer" in chunk and chunk["answer"]:
                        full_response += chunk["answer"]
                        yield chunk["answer"]

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                st.session_state["last_context"] = context_docs

            try:
                st.write_stream(stream_response())

                if st.session_state.get("last_context"):
                    with st.expander("📚 Retrieved passages"):
                        for i, doc in enumerate(st.session_state["last_context"], 1):
                            source = doc.metadata.get("source", "Unknown")
                            st.markdown(
                                f"**Source {i}** (`{source}`)\n\n{doc.page_content[:600]}..."
                            )
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")

st.caption("Built with LangChain + OpenAI + Pinecone • Index: chatwithpdf-1536")