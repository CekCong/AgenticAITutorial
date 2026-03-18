from dotenv import load_dotenv
from pydantic import BaseModel        # for data validation
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  # Claude models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_tool,pinecone_tool #import from tools.py
import streamlit as st



load_dotenv()   # Load env var from .env file

#data schema for the response
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
#llm = ChatOpenAI(model = "gpt-5.2")             #OpenAI model
llm = ChatAnthropic(model = "claude-3-haiku-20240307")   #Claude model
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#create agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a chatbot that will help answer any question the users ask.
            Answer the user query and use neccessary tools to help you. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# tools = [search_tool, wiki_tool, save_tool,pinecone_tool] #tool list
tools = [pinecone_tool,save_tool]

agent_graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt,
    response_format=ResearchResponse,
)



# Streamlit 
st.title("🔎 Search agent")

query = st.text_input("Ask a question")

if st.button("Search"):

    with st.spinner("Thinking..."):

        raw_response = agent_graph.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        try:
            structured_response = raw_response.get("structured_response")

            if structured_response is None:
                messages = raw_response.get("messages", [])
                last_message = messages[-1] if messages else None
                output_text = getattr(last_message, "content", None)
                st.write(output_text)
            else:
                st.subheader("Topic")
                st.write(structured_response.topic)

                st.subheader("Summary")
                st.write(structured_response.summary)

                st.subheader("Sources")
                for s in structured_response.sources:
                    st.write("-", s)

                st.subheader("Tools Used")
                for t in structured_response.tools_used:
                    st.write("-", t)

        except Exception as e:
            st.error(f"Error parsing response: {e}")
            st.write(raw_response)


