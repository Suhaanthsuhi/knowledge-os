from neo4j import GraphDatabase

from app.extraction.schema import KnowledgeGraph


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

    def verify_connectivity(self):
        return self.driver.verify_connectivity()

    def insert_graph(self, graph: KnowledgeGraph):
        with self.driver.session() as session:
            session.execute_write(
                self._insert_graph,
                graph,
            )

    @staticmethod
    def _insert_graph(tx, graph: KnowledgeGraph):

        # Insert entities
        for entity in graph.entities:
            tx.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                """,
                name=entity.name,
                type=entity.type,
            )

        # Insert relationships
        for relationship in graph.relationships:
            tx.run(
                """
                MATCH (source:Entity {name: $source})
                MATCH (target:Entity {name: $target})
                MERGE (source)-[r:RELATES_TO {type: $relationship}]->(target)
                """,
                source=relationship.source,
                target=relationship.target,
                relationship=relationship.relationship,
            )

    def find_entity_candidates(
        self,
        name: str,
        entity_type: str,
        limit: int = 10,
    ):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.type = $type
                AND toLower(e.name) CONTAINS toLower($name)
                RETURN e.name AS name, e.type AS type
                LIMIT $limit
                """,
                name=name,
                type=entity_type,
                limit=limit,
            )

            return [
                {
                    "name": record["name"],
                    "type": record["type"],
                }
                for record in result
            ]

    def close(self):
        self.driver.close()


__all__ = ["Neo4jClient"]