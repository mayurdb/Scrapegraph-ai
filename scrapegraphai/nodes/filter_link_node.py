"""
FilterLinkNode Module
"""

# Imports from standard library
from typing import List, Optional
from tqdm import tqdm
from bs4 import BeautifulSoup


# Imports from Langchain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel

# Imports from the library
from .base_node import BaseNode


class FilterLinkNode(BaseNode):
    """
    A node that filters all the relevant links to the prompt from the input links.
    Uses LLM for the filtering

    Attributes:
        llm_model: An instance of the language model client used for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "FilterLinks".
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[dict] = None,
                 node_name: str = "FilterLinks"):
        super().__init__(node_name, "node", input, output, 1, node_config)

        self.llm_model = node_config["llm_model"]
        self.verbose = False if node_config is None else node_config.get(
            "verbose", False)

    def execute(self, state: dict) -> dict:
        """
        Generates a list of relevant links to the task in prompt from the given input links.

        Args:
            state (dict): The current state of the graph. The input keys will be used to fetch the
                            correct data types from the state.

        Returns:
            dict: The updated state with the output key containing the list of links.

        """

        if self.verbose:
            print(f"--- Executing {self.node_name} Node ---")

        # Interpret input keys based on the provided input expression
        input_keys = self.get_input_keys(state)

        # Fetching data from the state based on the input keys
        input_data = [state[key] for key in input_keys]
        user_prompt = input_data[0]
        links = input_data[1]

        output_parser = JsonOutputParser()

        template_link_filter = """
        You are a website scraper and you have just scraped the
        following links from a webpage
        You are now asked to find all the relevant links that could should be 
        checked to accomplish the task mentioned in the instruction.\n 
        Original instruction: {user_prompt}
        """

        prompt = PromptTemplate(
            template=template_no_chunks,
            input_variables=["question"],
            partial_variables={"user_prompt": user_prompt,
                            },
        )

        answer = prompt.invoke()


        # Update the state with the generated answer
        state.update({self.output[0]: answer})
        return state
