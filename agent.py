from dotenv import load_dotenv
import os

import uuid
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.tools import tool


backend = StateBackend

@tool(parse_docstring = True)
def search_documentation(query):
    """
    
    """