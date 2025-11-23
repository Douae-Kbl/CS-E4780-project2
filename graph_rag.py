import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. 
    A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. 
    The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _():
    #rag_cache.clear()
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="What category of prizes has the biggest amount of winners", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager)[0]

    if result!={}:
        query = result['query']
        answer = result['answer'].response
    return answer, query


@app.cell
def _(answer, mo, query):
    mo.hstack([mo.md(f"""### Query\n```{query}```"""), mo.md(f"""### Answer\n{answer}""")])
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
        few_shot_examples: str = dspy.InputField()
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


    class RepairQuery(dspy.Signature):
        """
        - Use the provided question, the failing Cypher query, the EXPLAIN results, and the error. Generate a corrected query valid for   Kuzu.

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
        wrong_query: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        few_shot_examples: str = dspy.InputField()
        error: str = dspy.InputField()
        repaired_query: Query = dspy.OutputField()
    return AnswerQuestion, PruneSchema, RepairQuery, Text2Cypher


@app.cell
def _(BAMLAdapter, OPENROUTER_API_KEY, dspy):
    # Using OpenRouter. Switch to another LLM provider as needed
    lm = dspy.LM(
        model="openrouter/google/gemini-2.0-flash-001",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
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
        to: Node = Field(alias="from", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return GraphSchema, Query


@app.cell
def _(Chroma, OPENROUTER_API_KEY, OpenAIEmbeddings, chromadb, csv):
    class FewShotRetrieval():        
            def __init__(self, k: int, data_path: str = "data/generate_examples/nobel_questions_queries.csv"):
                self.k = k
                self.data_path = data_path

                self.embeddings = OpenAIEmbeddings(
                    model = "text-embedding-3-large",
                    api_key = OPENROUTER_API_KEY,
                    base_url = "https://openrouter.ai/api/v1"
                )
                self.client = chromadb.PersistentClient(path="./chroma_langchain_db")

                self.vector_store = Chroma(
                    collection_name = "nobel_examples",
                    embedding_function = self.embeddings,
                    client = self.client,
                    persist_directory = "./chroma_nobel_db"
                )

                if self.vector_store._collection.count() == 0:
                    self.load_examples()

            def load_examples(self):
                q = []
                metadatas = []
                ids = []
                with open(self.data_path, 'r', encoding='utf-8') as data:
                    reader = csv.DictReader(data)
                    for i, row in enumerate(reader):
                        question = row['question']
                        cypher = row['cypher']
                        q.append(question)
                        metadatas.append({'question': question, 'cypher': cypher})
                        ids.append(f"{i}")

                self.vector_store.add_texts(
                    texts=q,
                    metadatas=metadatas,
                    ids=ids
                )

            def retrieve_examples(self, question: str) -> str:
                k_shots_examples = self.vector_store.similarity_search(question, k=self.k)

                formatted_examples = []
                for example in k_shots_examples:
                    question = example.metadata['question']
                    query = example.metadata['cypher']
                    formatted_examples.append(f"Question: {question}\nCypher: {query}")

                return "\n\n".join(formatted_examples)

    return (FewShotRetrieval,)


@app.cell
def _(re):
    class RuleBasedProcessor():
        def __init__(self, query: str):
            self.query=query


        def equality_repl(self,match):
            prop = match.group(1)
            val = match.group(2)
            return f"lower({prop}) CONTAINS lower('{val}')"

        def lowercase_comp(self, query):
            # Ensure lowercase on CONTAINS comparisons
            contains_pattern = r"([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)\s+CONTAINS\s+['\"]([^'\"]+)['\"]"
            query = re.sub(contains_pattern, self.equality_repl, query)

            # Ensure lowercase on equality comparisons, transform to CONTAINS
            # obj.property = 'Value'  --> lower(obj.property) CONTAINS lower('Value')
            equality_pattern = r"([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)\s*=\s*['\"]([^'\"]+)['\"]"
            query = re.sub(equality_pattern, self.equality_repl, query)


            return query

        def get_label(self, item, query):
            print("item causing problem",item)
            # Regex pattern: (var:Label)
            item_escaped=re.escape(item)
            pattern = rf"\(\s*{item_escaped}\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\)"
            m = re.search(pattern, query)

            if m:
                return m.group(1)   # the label

            return ""


        def property_projection(self, query):
            # Get return clause
            m = re.search(r"RETURN\s+(.+?)(?=\s+ORDER BY|\s+LIMIT|\s+SKIP|$)", query, flags=re.IGNORECASE)
            if not m:
                return query
            print("query to check",query)
            projection = m.group(1)
            items = [i.strip() for i in projection.split(",")]
            new_items = []

            for item in items:
                # already has a property
                if "(" in item or ")" in item:
                    new_items.append(item)
                    continue
                if "." in item:
                    item_proprety=item.split(".")[0]

                    cur_property = item.split(".")[1]
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item_proprety.strip()):
                            new_items.append(item)
                            continue

                    label = self.get_label(item_proprety, query)

                    # scholar already has property of name but isn't 'knownName'
                    if label.lower() == "scholar" and "name" in cur_property.lower():
                        item = item.replace(cur_property, "knownName")


                    new_items.append(item)
                    continue

                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.strip()):
                        new_items.append(item)
                        continue
                # doesn' t have a property
                label = self.get_label(item, query)

                if label.lower() == "scholar":
                    new_items.append(f"{item}.knownName")
                elif label.lower() == "prize":
                    new_items.append(f"{item}.category")
                elif label.lower=="institution" or label.lower=="continent" or label.lower=="country" or label.lower=="city":
                    new_items.append(f"{item}.name")
                else:
                    new_items.append(f"{item}")

            new_return = "RETURN " + ", ".join(new_items)
            return query[:m.start()] + new_return + query[m.end():]


        def process_query(self):
            if not self.query:
                return self.query

            # Remove newlines and extra spaces
            query = ' '.join(self.query.split())

            # Apply lowercase transformations
            query = self.lowercase_comp(query)

            # Ensure proper property projection
            query = self.property_projection(query)


            return query


    return (RuleBasedProcessor,)


@app.cell
def _(OrderedDict):

    class LRUCache:

        def __init__(self, capacity):
            self.cache = OrderedDict()
            self.capacity = capacity

        def get(self, question):
            if question not in self.cache:
                return None
            else:
                self.cache.move_to_end(question)
                return self.cache[question]

        def put(self, question , answer):
            self.cache[question] = answer
            self.cache.move_to_end(question)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last = False)
        def clear(self):
            self.cache.clear()

    rag_cache=LRUCache(capacity=100)


    return (rag_cache,)


@app.cell
def _(
    AnswerQuestion,
    Any,
    FewShotRetrieval,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    RepairQuery,
    RuleBasedProcessor,
    Text2Cypher,
    dspy,
    hashlib,
    rag_cache,
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
            self.few_shots_retrieval= FewShotRetrieval(k=3)
            self.repair_query=dspy.ChainOfThought(RepairQuery)
            self.postprocess_query= None
            self.pipeline_latencies={}

        def get_cypher_query(self, question: str, input_schema: str,i: int, errors=None, wrong_query=None, mode: str="generate") -> Query:
            start_prune=time.perf_counter()
            prune_result = self.prune(question=question, input_schema=input_schema)
            self.pipeline_latencies[f"prune_{i}"]=time.perf_counter()-start_prune

            start_few_shots=time.perf_counter()
            k_shot_examples= self.few_shots_retrieval.retrieve_examples(question=question)
            self.pipeline_latencies[f"few_shots_{i}"]=time.perf_counter()-start_few_shots

            schema = prune_result.pruned_schema
            if mode=="generate":
                print(k_shot_examples)
                start_t2c=time.perf_counter()
                text2cypher_result = self.text2cypher(question=question, input_schema=schema,few_shot_examples=k_shot_examples)
                self.pipeline_latencies[f"gen_t2c_{i}"]=time.perf_counter()-start_t2c

                cypher_query = text2cypher_result.query
            elif mode=="repair":
                start_rep=time.perf_counter()
                repair_results= self.repair_query(question=question, wrong_query=wrong_query, input_schema=schema, few_shot_examples=k_shot_examples, error=errors)
                self.pipeline_latencies[f"rep_t2c_{i}"]=time.perf_counter()-start_rep
                cypher_query=repair_results.repaired_query
            return cypher_query


        def validate_query(self,db_manager,query):
            try:
                # Run the query on the database
                result = db_manager.conn.execute(f"EXPLAIN {query}")
                print(f"query{query}, passed explain")
                return True, None
            except RuntimeError as e:
                print(f"Error running query: {e}")
                return False, str(e)

        def run_query(
            self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
        ) -> tuple[str, list[Any] | None]:
            """
            Run a query synchronously on the database.
            """
            start_get_query=time.perf_counter()
            result = self.get_cypher_query(question=question, input_schema=input_schema,i=0)
            self.pipeline_latencies["get_cypher_query"]=time.perf_counter()-start_get_query

            self.postprocess_query= RuleBasedProcessor(result.query)

            start_postprocess_query=time.perf_counter()
            query = self.postprocess_query.process_query()
            self.pipeline_latencies["postprocess_query"]=time.perf_counter()-start_postprocess_query

            start_validate_query=time.perf_counter()
            status,errors=self.validate_query(db_manager,query)
            self.pipeline_latencies["validate_query"]=time.perf_counter()-start_validate_query

            max_tries=6
            i=0
            while (not status) and i<max_tries:
                print("inside repair loop")
            
                start_rep_query=time.perf_counter()
                result = self.get_cypher_query(question=question, input_schema=input_schema, wrong_query=query, errors=errors, mode="repair", i=i)
                self.pipeline_latencies[f"rep_query_{i}"]=time.perf_counter()-start_rep_query
            
                self.postprocess_query= RuleBasedProcessor(result.query)
                start_postprocess_query=time.perf_counter()
                query = self.postprocess_query.process_query()
                self.pipeline_latencies[f"rep_postprocess_query_{i}"]=time.perf_counter()-start_postprocess_query

                start_validate_query=time.perf_counter()
                status,errors=self.validate_query(db_manager,query)
                self.pipeline_latencies[f"rep_validate_query_{i}"]=time.perf_counter()-start_validate_query
                i+=1
            try:
                # Run the query on the database
                start_exec_query=time.perf_counter()
                print(f"executable query{query}")
                result = db_manager.conn.execute(query)
                self.pipeline_latencies["exec_query"]=time.perf_counter()-start_exec_query
            
                results = [item for row in result for item in row]
                print(f"final result {results}")
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
            start_cached=time.perf_counter()
            question_hashed = hashlib.sha256(question.encode("utf-8")).hexdigest()
            answer_cached=rag_cache.get(question_hashed)
        
            rag.pipeline_latencies["cache_retrieval"]=time.perf_counter()-start_cached

            if answer_cached is not None:
                response= answer_cached
            else:
                response = rag(db_manager=db_manager, question=question, input_schema=schema)
                start_caching=time.perf_counter()
                question_hashed = hashlib.sha256(question.encode("utf-8")).hexdigest()
                rag_cache.put(question_hashed,response)
                rag.pipeline_latencies["caching"]=time.perf_counter()-start_caching
            rag.pipeline_latencies["full_run"]=time.perf_counter()-start_full_run

            results.append(response)
            print(rag.pipeline_latencies)
        return results

    return (run_graph_rag,)


@app.cell
def _():
    import marimo as mo
    import re
    import os
    from textwrap import dedent
    from typing import Any
    from collections import OrderedDict

    import dspy
    import kuzu
    from dotenv import load_dotenv
    from dspy.adapters.baml_adapter import BAMLAdapter
    from pydantic import BaseModel, Field
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import chromadb
    import time
    import csv
    import hashlib

    load_dotenv()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Chroma,
        Field,
        OPENROUTER_API_KEY,
        OpenAIEmbeddings,
        OrderedDict,
        chromadb,
        csv,
        dspy,
        hashlib,
        kuzu,
        mo,
        re,
        time,
    )


if __name__ == "__main__":
    app.run()
