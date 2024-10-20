import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage , SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import time
from langchain_mistralai.chat_models import ChatMistralAI
import base64
from io import BytesIO
from mistralai import Mistral


load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
mistral_api_key = os.getenv('MISTRAL_API_KEY')

llm2 = ChatMistralAI(model_name='open-mistral-7b')
client = Mistral(api_key=mistral_api_key)
model = "pixtral-12b-2409"

search= TavilySearchResults(max_results=3)
tools = [search]

prompt1 = """
You are an expert landmark identifier. Based on the given Image, detect its name. 
Output only the name nothing else.

"""

memory = MemorySaver()

st.title('AI Tour Guide')

uploaded_image = st.file_uploader("Upload an image of a monument", type=["jpg", "jpeg", "png"])

if uploaded_image:
    
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  
    img_str = base64.b64encode(buffered.getvalue()).decode()

    if image.mode == "RGBA":
        image = image.convert("RGB")

    
    
    def identify_monument():
        messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt1
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{img_str}" 
                                }
                            ]
                        }
                    ]
        chat_response = client.chat.complete(
        model=model,
        messages=messages
        )

        result = chat_response.choices[0].message.content

        return result
    
    monument_name = identify_monument()
    st.write(f"Identified Monument: **{monument_name}**")
    
    
    if monument_name:
        time.sleep(5)
        agent_executor = create_react_agent(llm2 , tools , checkpointer=memory)

        config = {'configurable': {'thread_id' : 'adv1'}}


        prompt_template = PromptTemplate(
        input_variables=["monument_name"],
        template="Tell me more about {monument_name} and its historical significance."
        )

        prompt2 = prompt_template.format(monument_name=monument_name)

        response = agent_executor.invoke(
        input={
            'messages': [HumanMessage(content=prompt2)]
        },
        config=config
        )
       
        st.write("### Monument Information:")
        st.write(response['messages'])


st.write("### Suggested Prompts:")
st.button("Tell me more about its architecture")
st.button("What are some famous events associated with it?")
st.button("Who built it and why?")
