from langchain_ollama import ChatOllama
import streamlit as st

llm = ChatOllama(
    model="llama3",
    temperature=0.8,
    num_predict=256,
    # other params ...
)

st.title("Llama3-like clone")
ques = 'What would be a cool name for a personal photo management agent ?'

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "llama_model" not in st.session_state:
    st.session_state["llama_model"] = "llama3"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = llm.invoke([
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ])

        # stream = llm.chat.completions.create(
        #     model=st.session_state["llama_model"],
        #     messages=,
        #     stream=True,
        # )
        response = st.write(stream.content)
    st.session_state.messages.append({"role": "assistant", "content": response})