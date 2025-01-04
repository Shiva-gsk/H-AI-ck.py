import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "./model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("./model")

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

def generate_answer(query, context, model, tokenizer, device):
    # Format the input
    input_text = f'''{context} 
    You are study companion. Answer this Question with bullet points based on above context make a good explanation to your answer: {query}'''

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for the new token
    #Tokenize input with attention mask and padding
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,  # Ensures inputs are padded
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate output with attention mask and pad_token_id
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=1024,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id  # Ensure proper handling of padding
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def handle_query(query, device):
    # Retrieve relevant chunks
    relevant_chunks = db.similarity_search_with_relevance_scores(query, k=3)
    context_text = " ".join([doc.page_content for doc, _score in relevant_chunks])
    
    # Generate an answer
    answer = generate_answer(query, context_text, model, tokenizer, device)
    return answer

interface = gr.Interface(
    fn=handle_query, 
    inputs=gr.Textbox(lines=10, placeholder="Enter Text here..", label="Input Text"), 
    outputs= gr.Textbox(label="Answer"),
    title="Question Answering System",
    )

interface.launch()




