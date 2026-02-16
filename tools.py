#to query Wiki and perform search on web
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from datetime import datetime

def save_to_txt(data: str, filename: str = "response_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Response by AIBOT ---\nTime of response: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = StructuredTool.from_function(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun() # allows us to perform searches on the web

def web_search(query: str) -> str:
    return search.run(query)

search_tool = StructuredTool.from_function(
    name="search",
    func=web_search,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
