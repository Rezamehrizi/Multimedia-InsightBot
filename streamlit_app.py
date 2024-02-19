import streamlit as st
import os
from langchain.llms import OpenAI  # Note: Still importing OpenAI for model loading
from langchain.llms import TransformersLLM
from langchain.agents import ConversationalAgent
from langchain.llms.pipelines import QuestionAnsweringPipeline


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
        
        
st.title(":red[QuizTube] — Watch. Learn. Quiz. 🧠", anchor=False)
st.write("""
Ever watched a YouTube video and wondered how well you understood its content? Here's a fun twist: Instead of just watching on YouTube, come to **QuizTube** and test your comprehension!

**How does it work?** 🤔
1. Paste the YouTube video URL of your recently watched video.
2. Enter your [OpenAI API Key](https://platform.openai.com/account/api-keys).

⚠️ Important: The video **must** have English captions for the tool to work.

Once you've input the details, voilà! Dive deep into questions crafted just for you, ensuring you've truly grasped the content of the video. Let's put your knowledge to the test! 
""")

with st.expander("💡 Video Tutorial"):
    with st.spinner("Loading video.."):
        st.video("https://youtu.be/yzBr3L2BIto", format="video/mp4", start_time=0)

with st.form("user_input"):
    YOUTUBE_URL = st.text_input("Enter the YouTube video link:", value="https://youtu.be/bcYwiwsDfGE?si=qQ0nvkmKkzHJom2y")
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", placeholder="sk-XXXX", type='password')
    submitted = st.form_submit_button("Craft my quiz!")


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


# Function for generating response using Langchain and Hugging Face
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."

    # Load Hugging Face model using TransformersLLM
    model_name = "facebook/bart-base"  # Replace with your desired Hugging Face model
    llm = TransformersLLM(model_name=model_name)

    # Create a question answering pipeline
    pipeline = QuestionAnsweringPipeline(llm=llm)

    # Construct the prompt for question answering
    prompt = f"Answer the following question based on the given context:\nContext:\n{string_dialogue}\nQuestion:\n{prompt_input}"

    # Generate the response using the pipeline
    response = pipeline(prompt=prompt)

    return response.answer
# # Function for generating response using OpenAI
# def generate_llama2_response(prompt_input):
#     string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
#     for dict_message in st.session_state.messages:
#         if dict_message["role"] == "user":
#             string_dialogue += "User: " + dict_message["content"] + "\n\n"
#         else:
#             string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

#     openai.api_key = os.environ.get("OPENAI_API_KEY")  # Set your OpenAI API key
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo",
#         prompt=f"{string_dialogue} {prompt_input} Assistant:",
#         temperature=temperature,
#         max_tokens=max_length,
#         n=1,
#         stop=None,
#         presence_penalty=0.0,
#         frequency_penalty=0.0,
#         best_of=1,
#     )
#     return response.choices[0].text.strip()

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
