from langchain_groq import ChatGroq

from app.config import settings
from app.extraction.schema import KnowledgeGraph


class GraphExtractor:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=settings.groq_api_key,
        )

        self.structured_llm = self.llm.with_structured_output(
            KnowledgeGraph
        )

    def extract(self, text: str) -> KnowledgeGraph:
        result = self.structured_llm.invoke(
            f"""
            Extract a knowledge graph from the following text.

            Identify important entities such as:
            - people
            - companies
            - organizations
            - locations
            - products
            - technologies
            - concepts

            Identify meaningful relationships between entities.

            Return only information explicitly supported by the text.

            Text:
            {text}
            """
        )

        return result


__all__ = ["GraphExtractor"]