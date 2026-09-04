from neo4j import GraphDatabase
from app.config import settings


class Neo4jClient:

    def __init__(self, uri, username, password):
        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

    @property
    def driver(self):
        return self._driver

    def verify_connectivity(self):
        return self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    def create_entity(self, name: str, entity_type: str):
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        """

        with self.driver.session() as session:
            session.run(query, name=name)

    def create_relationship(
        self,
        source: str,
        relationship: str,
        target: str,
    ):
        query = f"""
        MATCH (source {{name: $source}})
        MATCH (target {{name: $target}})
        MERGE (source)-[:{relationship}]->(target)
        """

        with self.driver.session() as session:
            session.run(
                query,
                source=source,
                target=target,
            )


__all__ = ["Neo4jClient"]