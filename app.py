import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.title("Math Riddles Generator (GPT-2)")
st.write("Enter a prompt to generate math riddles:")

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_name = "izaanishaq/math_riddle-gpt2"  
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
        st.write("Using CUDA")
    else:
        st.write("Using CPU")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()


def generate_riddles(prompt="", num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=100,
        top_k=50,
        top_p=0.92,
        temperature=0.7,
        repetition_penalty=1.2,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    riddles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return riddles

user_prompt = st.text_input("Prompt", "Here is a math riddles:")

if st.button("Generate Riddle"):
    with st.spinner("Generating riddle..."):
        riddle = generate_riddles(user_prompt, num_return_sequences=1)[0]
    st.subheader("Generated Riddle:")
    st.markdown(f"**{riddle}**")