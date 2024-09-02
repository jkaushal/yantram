import streamlit as st

from src.main.utils.utils import create_docs, create_embeddings_load_data, push_to_pinecone, similar_docs, get_summary
from langchain_community.llms import Ollama

from utils import convert_to_base64, convert_to_html

from utils import *
import uuid

# Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''


def main():
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...💁")
    st.subheader("I can help you in resume screening process")
    # job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...", key="1")
    # document_count = st.text_input("No.of 'RESUMES' to return", key="2")

    # Upload the Resumes (pdf files)
    images = st.file_uploader("Upload an image to chat about", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    # assert max number of images, e.g. 7
    images_b64 = []
    submit = st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):
            question = st.chat_input("Ask a question about the image(s)")
            print('Hello QUESTION')
            print(question)
            print('Hello QUESTION')
            # Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id'] = uuid.uuid4().hex

            # Create a documents list out of all the user uploaded pdf files

            # Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :" + str(len(images)))

            # Create embeddings instance
            # embeddings = create_embeddings_load_data()
            for image in images:
                image_b64 = convert_to_base64(image)
                images_b64.append(image_b64)
            llm = Ollama(model='llava')
            llm_with_image_context = llm.bind(images=image_b64)
            res = llm_with_image_context.invoke([])
            st.write(res)

            # Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            # For each item in relavant docs - we are displaying some info of it on the UI
        st.success("Hope I was able to save your time❤️")


# Invoking main function
if __name__ == '__main__':
    main()