import streamlit as st
import os
import openai  # Import the OpenAI library

# App title 
st.set_page_config(page_title=" Llama 2 Chatbot (with OpenAI)")

# Replicate Credentials (sidebar)
with st.sidebar:
    st.title(' Llama 2 Chatbot')
    st.write('This chatbot is created using OpenAI\'s text-davinci-003 model.')
    st.subheader('API Key')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
    else:
        openai_api_key = st.text_input('Enter OpenAI API key:', type='password')
        os.environ['OPENAI_API_KEY'] = openai_api_key

    # Add a collapsible section for parameters
    with st.expander("Parameters"):
        temperature = st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        max_length = st.slider('max_length', min_value=32, max_value=128, value=120, step=8)
        
        
        

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating response using OpenAI
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    openai.api_key = os.environ.get("OPENAI_API_KEY")  # Set your OpenAI API key
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f"{string_dialogue} {prompt_input} Assistant:",
        temperature=temperature,
        max_tokens=max_length,
        n=1,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        best_of=1,
    )
    return response.choices[0].text.strip()

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
