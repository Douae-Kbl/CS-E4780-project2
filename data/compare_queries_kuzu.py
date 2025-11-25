#!/usr/bin/env python3
"""
Run queries from `data/comparison_queries.csv` against the local Kuzu DB (`nobel.kuzu`) and
decide whether RAG / initial queries match the gold `cypher` results.

Output: `data/comparison_results.csv` with results and boolean match flags.
"""
from __future__ import annotations

import ast
import csv
import sys
from collections import Counter
from typing import Any

import pandas as pd
import kuzu


def connect(db_path: str = "../nobel.kuzu") -> kuzu.Connection:
    db = kuzu.Database(db_path, read_only=True)
    return kuzu.Connection(db)


def run_query(conn: kuzu.Connection, query: str) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    """Run a query on Kuzu. Return (rows, error_message).

    Rows are returned as a list of tuples. On error, rows is None and error_message contains text.
    """
    if query is None:
        return None, "no query"

    q = str(query).strip()
    if not q or q.lower().startswith("no valid"):
        return None, "no query"

    try:
        # Validate with EXPLAIN first (helps catching syntax problems early)
        try:
            conn.execute(f"EXPLAIN {q}")
        except RuntimeError:
            # EXPLAIN can fail for some valid queries depending on Kuzu; continue to execute
            pass

        result = conn.execute(q)
        rows = [tuple(row) for row in result]
        return rows, None
    except RuntimeError as e:
        return None, str(e)


def canonicalize_rows(rows: list[tuple[Any, ...]] | None) -> Counter:
    """Convert rows to a multiset of canonical string tuples for robust comparison."""
    if rows is None:
        return Counter()

    def canon_val(v: Any) -> str:
        # Basic canonicalization: None -> '', bytes -> decode, others -> str stripped
        if v is None:
            return ""
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8").strip()
            except Exception:
                return repr(v)
        return str(v).strip()

    canon = [tuple(canon_val(x) for x in row) for row in rows]
    return Counter(canon)


def compare_results(a: list[tuple[Any, ...]] | None, b: list[tuple[Any, ...]] | None) -> bool:
    """Return True if result sets/multisets match.

    We canonicalize rows and compare Counters to handle ordering and duplicate rows.
    """
    return canonicalize_rows(a) == canonicalize_rows(b)


def main(csv_path: str = "data/comparison_queries.csv", db_path: str = "../nobel.kuzu") -> int:
    df = pd.read_csv(csv_path)
    conn = connect(db_path)

    out_rows = []

    total = len(df)
    rag_match_count = 0
    init_match_count = 0

    for idx, row in df.iterrows():
        qid = int(row.get("", idx)) if "" in row else idx
        question = row.get("question") if "question" in row else ""

        # columns present in your CSV: 'cypher', 'rag_query', 'initial_query'
        gold_q = row.get("cypher") if "cypher" in row else None
        rag_q = row.get("rag_query") if "rag_query" in row else None
        init_q = row.get("initial_query") if "initial_query" in row else None

        gold_rows, gold_err = run_query(conn, gold_q)
        rag_rows, rag_err = run_query(conn, rag_q)
        init_rows, init_err = run_query(conn, init_q)

        rag_matches = False
        init_matches = False

        if gold_rows is not None and rag_rows is not None:
            rag_matches = compare_results(gold_rows, rag_rows)
        if gold_rows is not None and init_rows is not None:
            init_matches = compare_results(gold_rows, init_rows)

        if rag_matches:
            rag_match_count += 1
        if init_matches:
            init_match_count += 1

        out_rows.append(
            {
                "id": idx,
                "question": question,
                "gold_query": gold_q,
                "rag_query": rag_q,
                "initial_query": init_q,
                "gold_result": repr(gold_rows) if gold_rows is not None else "<ERROR>",
                "rag_result": repr(rag_rows) if rag_rows is not None else "<ERROR>",
                "initial_result": repr(init_rows) if init_rows is not None else "<ERROR>",
                "rag_matches_gold": rag_matches,
                "initial_matches_gold": init_matches,
                "gold_error": gold_err,
                "rag_error": rag_err,
                "initial_error": init_err,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = "data/comparison_results.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Processed {total} rows. RAG matches: {rag_match_count}. Initial matches: {init_match_count}.")
    print(f"Results written to: {out_path}")
    return 0


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/comparison_queries.csv"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "../nobel.kuzu"
    raise SystemExit(main(csv_path=csv_path, db_path=db_path))
