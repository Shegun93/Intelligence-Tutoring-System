import streamlit as st
import torch
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers.utils import is_flash_attn_2_available
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore

# Import your local modules
from Retrieval_system import Retrieval
from pincone import Pincone_vectorStore

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. SYSTEM INITIALIZATION ---
@st.cache_resource
def load_socratic_engine():
    """Loads models and retrieval objects into GPU memory once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Retrieval
    index_obj = Pincone_vectorStore() 
    embeddings = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2", 
        model_kwargs={"device": device}
    )
    vector_store = PineconeVectorStore(embedding=embeddings, index=index_obj)
    retrieval_obj = Retrieval(
        device=device, 
        index=index_obj, 
        Embeddings=embeddings, 
        vector_store=vector_store
    )

    # Setup LLM
    MODEL_ID = "/home/shegun93/ITS/Fine-tunning/nairsV1"
    attn_impl = "flash_attention_2" if (is_flash_attn_2_available() and (torch.cuda.get_device_capability(0)[0] >= 8)) else "sdpa"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        attn_implementation=attn_impl,
        device_map="auto"
    )
    
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256, # Increased for more conversational depth
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    return retrieval_obj, llm

# --- 2. UTILS ---
def parse_mcq(doc_text):
    lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
    question, options = None, {}
    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif len(line) > 2 and line[1] == "." and line[0] in ["A", "B", "C", "D"]:
            options[line[0]] = line[2:].strip()
    return question, options

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Personalized Socratic Tutor", layout="wide")
st.title("🎓 Socratic AI Tutor")
st.markdown("---")

# Load Backend
with st.spinner("Waking up the tutor..."):
    retrieval_system, llm_engine = load_socratic_engine()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "internal_history" not in st.session_state:
    st.session_state.internal_history = [] 
if "active_context" not in st.session_state:
    st.session_state.active_context = None

# --- 4. SIDEBAR: TOPIC SELECTION ---
with st.sidebar:
    st.header("Learning Session")
    topic = st.text_input("What do you want to learn today?", placeholder="e.g. Quantum Physics")
    
    if st.button("Start Lesson"):
        retriever, _ = retrieval_system.get_retrieval()
        docs = retriever.get_relevant_documents(topic)
        if docs:
            q, opts = parse_mcq(docs[0].page_content)
            st.session_state.active_context = {
                "question": q,
                "options": opts,
                "full_text": docs[0].page_content
            }
            # Clear previous chat
            st.session_state.messages = []
            st.session_state.internal_history = []
            st.rerun()
        else:
            st.error("No knowledge found on this topic.")

# --- 5. THE CONVERSATION ---
if st.session_state.active_context:
    ctx = st.session_state.active_context
    
    # Display the "Springboard" Question
    with st.expander("Current Lesson Challenge", expanded=True):
        st.info(f"**Conceptual Challenge:** {ctx['question']}")
        for k, v in ctx['options'].items():
            st.write(f"**{k}**: {v}")

    # Show Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Share your thoughts or ask a question..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build Socratic Prompt
        # This prompt forces the model to be a teacher, not an answer key.
        socratic_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a Personalized Socratic Tutor. Your goal is to lead the student to realize the truth through reasoning.\n"
            f"RULES:\n"
            f"1. NEVER reveal the answer or say 'The correct option is...'.\n"
            f"2. Use analogies and relatable examples to explain complex parts of the CONTEXT.\n"
            f"3. If the student is wrong or says 'I don't know', ask a simpler question to build their confidence.\n"
            f"4. If the student is correct, ask them WHY they think so to ensure it wasn't a guess.\n"
            f"5. Wrap strategy in <diagnosis> and student response in <hint>.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"CONTEXT: {ctx['full_text']}\n"
            f"TOPIC: {ctx['question']}\n"
            f"STUDENT SAYS: {user_input}\n"
            f"CONVERSATION LOG: {st.session_state.internal_history[-5:]}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"<diagnosis>"
        )

        with st.chat_message("assistant"):
            with st.spinner("Teaching..."):
                response = llm_engine.invoke(socratic_prompt)
                
                # Extract clean response
                hint_match = re.search(r'<hint>(.*?)</hint>', "<diagnosis>" + response, re.DOTALL)
                tutor_text = hint_match.group(1).strip() if hint_match else "That's an interesting perspective. Tell me more about why you think that?"
                
                # Extract diagnosis for the "Thinking Process" expander
                diag_match = re.search(r'<diagnosis>(.*?)</diagnosis>', "<diagnosis>" + response, re.DOTALL)
                
                st.markdown(tutor_text)
                if diag_match:
                    with st.expander("🔍 Tutor's Strategy"):
                        st.write(diag_match.group(1))

        # Update State
        st.session_state.messages.append({"role": "assistant", "content": tutor_text})
        st.session_state.internal_history.append(f"Student: {user_input}, Tutor: {tutor_text}")
else:
    st.info("👈 Enter a topic in the sidebar to begin your Socratic session.")