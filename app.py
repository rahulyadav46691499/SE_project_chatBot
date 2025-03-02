import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Replace with your actual Gemini API key - SECURITY WARNING!
GEMINI_API_KEY = "AIzaSyBe7tb7bcO7MNlUjbIqB0X6OFTwnbNDq7U"  # Store securely in a real app!


@st.cache_resource
def create_gemini_chatbot(prompt_template_string, llm_temperature=0.7):
    """
    Creates a chatbot using the Gemini API and Langchain.  Caches the LLM.

    Args:
        prompt_template_string: The prompt template to use.  Should include an `input` variable.
        llm_temperature: The temperature setting for the Gemini model.

    Returns:
        A function that takes a user input string and returns the chatbot's response.
    """

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",  # Corrected model name
                                 google_api_key=GEMINI_API_KEY,
                                 temperature=llm_temperature)

    # Create PromptTemplate
    prompt = PromptTemplate(template=prompt_template_string, input_variables=["input"])

    # Create LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    def chatbot(user_input):
        """
        The actual chatbot function that takes user input and returns a response.
        """
        return chain.run(user_input)

    return chatbot


def main():
    """
    Creates a Streamlit app with a Gemini-powered chatbot.
    """

    st.title("Gemini Chatbot")

    # Define a prompt template
    template = """You are a friendly chatbot.  When asked a question, you try to
            answer it. Your goal is to provide helpful and informative responses.

            User: {input}
            Chatbot:"""

    # Create the chatbot - do this outside the chat loop
    chatbot = create_gemini_chatbot(template)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the chatbot
        try:
            response = chatbot(prompt)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your API key and internet connection.")



if __name__ == "__main__":
    main()