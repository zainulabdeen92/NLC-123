from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
#  Put yoyur API key
OPENAI_API_KEY = ''

model_name = 'gpt-3.5-turbo'

llm = ChatOpenAI(model_name=model_name, temperature=0.3, openai_api_key=OPENAI_API_KEY)

def chatbot(info, history, message):
    print("history:", history)
    prompt_template = """
    You are a farming expert with specialized knowledge in assessing water quality for irrigation.\
    Also, some water quality parameters that the farmer uses\
    will be provided to you (do not ask the user for this information from user in chat) such as pH, hardness, solids, chloramines,\
    sulfate, conductivity, Organic_carbon, Trihalomethanes, Turbidity. There values are {info} respectively.\
    
    Once you have the information and chat history, your task is to give suggestions to farmers on what steps they should take to improve water quality,\
    based on the input parameters and crop type. Your responses should be precise and have a human conversational touch.\
    Do not generate responses until the farmer provides you with the necessary information.\
    
    Chat history: {history},

    User question: {message}
    
    If the user asks anything outside of this scope, display the message: I am a water quality assistant,\
    and I cannot provide answers outside of this scope.
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['info', 'history', 'message']
    )

 
    chain = LLMChain(llm=llm, prompt=PROMPT)

    response = chain.predict(info=info, history=history, message=message)

    return response
