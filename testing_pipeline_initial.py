import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="Which scholars won prizes in Physics were affiliated with University of Oxford?", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    df = pd.read_csv("data/generate_examples/nobel_questions_queries.csv")
    questions = df['question'].tolist()

    i = 0
    answers = []
    queries = []
    latencies = []

    # with mo.status.spinner(title="Generating answer...") as _spinner:
    #     results= run_graph_rag([question], db_manager)
    #     result=results[0]
    #     latencies=results[1]

    # query = result[0]['query']
    # answer = result[0]['answer'].response
    for i, question in enumerate(questions):
        with mo.status.spinner(title=f"Generating answer {i}..."):

            # IMPORTANT: wrap `question` in a LIST
            results = run_graph_rag([question], db_manager)
            result = results[0]

        if result != {}:
            if 'query' not in result[0].keys():
                query = "No valid query generated"
                answer = "No answer generated"
            else:
                query = result[0]['query']
                answer = result[0]['answer'].response
                
            queries.append(query)
            answers.append(answer)
            latencies.append(results[1])

    return answers, queries, questions, latencies


@app.cell
def _(answers, mo, queries):
    mo.hstack([mo.md(f"""### Query\n```{queries}```"""), mo.md(f"""### Answer\n{answers}""")])
    return

@app.cell
def _(answers, latencies, pd, queries, questions):
    # print results to file:
    df_out = pd.DataFrame({
        "question": questions,
        "generated_query": queries,
        "generated_answer": answers,
        "latencies": latencies
    })
    df_out.to_csv("data/test_results_initial_pipeline2.csv", index=False)
    return


@app.cell
def _(GraphSchema, Query, dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
            - The schema is a list of nodes and edges in a property graph.
            - The nodes are the entities in the graph.
            - The edges are the relationships between the nodes.
            - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema: GraphSchema = dspy.OutputField()


    class Text2Cypher(dspy.Signature):
        """
        Translate the question into a valid Cypher query that respects the graph schema.

        <SYNTAX>
        - When matching on Scholar names, ALWAYS match on the `knownName` property
        - For countries, cities, continents and institutions, you can match on the `name` property
        - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
        - Always strive to respect the relationship direction (FROM/TO) using the schema information.
        - When comparing string properties, ALWAYS do the following:
            - Lowercase the property values before comparison
            - Use the WHERE clause
            - Use the CONTAINS operator to check for presence of one substring in the other
        - DO NOT use APOC as the database does not support it.
        </SYNTAX>

        <RETURN_RESULTS>
        - If the result is an integer, return it as an integer (not a string).
        - When returning results, return property values rather than the entire node or relationship.
        - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
        - NO Cypher keywords should be returned by your query.
        </RETURN_RESULTS>
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        query: Query = dspy.OutputField()


    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question, the generated Cypher query and the context to answer the question.
        - If the context is empty, state that you don't have enough information to answer the question.
        - When dealing with dates, mention the month in full.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()
    return AnswerQuestion, PruneSchema, Text2Cypher


@app.cell
def _(BAMLAdapter, OPENROUTER_API_KEY, dspy):
    # Using OpenRouter. Switch to another LLM provider as needed
    lm = dspy.LM(
        model="openrouter/google/gemini-2.0-flash-001",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        cache=False,
    )
    dspy.configure(lm=lm, adapter=BAMLAdapter())
    return


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "ldbc_1.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]  # type: ignore
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]  # type: ignore
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:  # type: ignore
                    node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:  # type: ignore
                    edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")


    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")


    class Node(BaseModel):
        label: str
        properties: list[Property] | None


    class Edge(BaseModel):
        label: str = Field(description="Relationship label")
        from_: Node = Field(alias="from", description="Source node label")
        to: Node = Field(alias="to", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return GraphSchema, Query


@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    Text2Cypher,
    dspy,
    time,
):
    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self):
            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = dspy.ChainOfThought(Text2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
            self.pipeline_latencies={}

        def get_cypher_query(self, question: str, input_schema: str) -> Query:
            start_prune=time.perf_counter()
            prune_result = self.prune(question=question, input_schema=input_schema)
            self.pipeline_latencies["prune"]=time.perf_counter()-start_prune
        
            schema = prune_result.pruned_schema

            start_t2c=time.perf_counter()
            text2cypher_result = self.text2cypher(question=question, input_schema=schema)
            self.pipeline_latencies["gen_t2c"]=time.perf_counter()-start_t2c
        
            cypher_query = text2cypher_result.query
            return cypher_query

        def run_query(
            self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
        ) -> tuple[str, list[Any] | None]:
            """
            Run a query synchronously on the database.
            """
            start_get_query=time.perf_counter()
            result = self.get_cypher_query(question=question, input_schema=input_schema)
            self.pipeline_latencies["get_cypher_query"]=time.perf_counter()-start_get_query
        
            query = result.query
            try:
                # Run the query on the databas
                start_exec_query=time.perf_counter()
                result = db_manager.conn.execute(query)
                self.pipeline_latencies["exec_query"]=time.perf_counter()-start_exec_query

                results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"Error running query: {e}")
                self.pipeline_latencies["exec_query"]=time.perf_counter()-start_exec_query
                results = None
            return query, results

        def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            start_run_query = time.perf_counter()
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                self.pipeline_latencies["run_query_forward"]=time.perf_counter()-start_run_query

                return {}
            else:
                start_answer_gen=time.perf_counter()
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                self.pipeline_latencies["answer_gen"]=time.perf_counter()-start_answer_gen
            
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                self.pipeline_latencies["run_query_forward"]=time.perf_counter()-start_run_query
                return response

        async def aforward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response


    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager) -> list[Any]:
        start_full_run=time.perf_counter()
        schema = str(db_manager.get_schema_dict)
        rag = GraphRAG()
        # Run pipeline
        results = []
        for question in questions:
            response = rag(db_manager=db_manager, question=question, input_schema=schema)
            results.append(response)
        rag.pipeline_latencies["full_run"]=time.perf_counter()-start_full_run

        return results,rag.pipeline_latencies

    return (run_graph_rag,)


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import os
    from textwrap import dedent
    from typing import Any
    import time
    import pandas as pd

    import dspy
    import kuzu
    from dotenv import load_dotenv
    from dspy.adapters.baml_adapter import BAMLAdapter
    from pydantic import BaseModel, Field

    load_dotenv()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Field,
        OPENROUTER_API_KEY,
        dspy,
        kuzu,
        mo,
        time,
    )


if __name__ == "__main__":
    app.run()
