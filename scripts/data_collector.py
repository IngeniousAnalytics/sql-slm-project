"""
data_collector.py - Real-world Domain Dataset Generator (schema.json -> JSONL)

✅ Reads schema.json created by schema_extractor.py
✅ Generates domain SQL training data:
   - business rules filters (is_deleted=false, status='ACTIVE', tenant_id=1, etc.)
   - CTE analytics
   - Window functions
   - Multi-table joins (2–4 tables via FK paths)
✅ Writes JSONL live (progress visible & file grows in real time)

Usage:
    python scripts/data_collector.py --total-examples 100
"""

import os
import json
import argparse
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SCHEMA_FILE = os.getenv("SCHEMA_OUTPUT_PATH", "./data/schemas/schema.json")

DEFAULT_OUT_DIR = "./data/processed/training_data.json"
DEFAULT_TRAIN_FILE = "train_prompts.jsonl"
DEFAULT_VAL_FILE = "validation_prompts.jsonl"


# -----------------------------
# Console helpers
# -----------------------------
def log(msg: str):
    print(msg, flush=True)


def progress_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return "[----------]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {current}/{total}"


# -----------------------------
# IO helpers
# -----------------------------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_train_val(rows: List[Dict[str, str]], val_ratio: float = 0.1):
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio)) if rows else 0
    return rows[n_val:], rows[:n_val]


# -----------------------------
# Schema formatting
# -----------------------------
def normalize_table_ref(db_type: str, schema: str, table: str) -> str:
    if db_type == "sqlite":
        return table
    if schema and schema.lower() not in ("public", "dbo", "main"):
        return f"{schema}.{table}"
    return table


def schema_to_text(schema_json: Dict[str, Any], max_tables=12, max_cols=14) -> str:
    lines = []
    for t in schema_json.get("tables", [])[:max_tables]:
        tname = t.get("name")
        tschema = t.get("schema", "")
        lines.append(f"Table: {tschema}.{tname}" if tschema else f"Table: {tname}")

        for c in t.get("columns", [])[:max_cols]:
            lines.append(f"  - {c.get('name')} ({c.get('data_type')})")
        lines.append("")
    return "\n".join(lines).strip()


# -----------------------------
# Column heuristics
# -----------------------------
def col_exists(table: Dict[str, Any], col_name: str) -> bool:
    return any(c.get("name", "").lower() == col_name.lower() for c in table.get("columns", []))


def find_column_like(table: Dict[str, Any], keywords: List[str]) -> Optional[str]:
    cols = table.get("columns", [])
    for kw in keywords:
        for c in cols:
            if kw in c.get("name", "").lower():
                return c.get("name")
    return None


def find_id_column(table: Dict[str, Any]) -> Optional[str]:
    cols = table.get("columns", [])
    for c in cols:
        n = c.get("name", "").lower()
        if n == "id" or n.endswith("_id"):
            return c.get("name")
    return cols[0].get("name") if cols else None


def find_numeric_column(table: Dict[str, Any]) -> Optional[str]:
    numeric_types = ("int", "numeric", "decimal", "real", "double", "float")
    for c in table.get("columns", []):
        dt = str(c.get("data_type", "")).lower()
        if any(nt in dt for nt in numeric_types):
            return c.get("name")
    return None


def find_text_column(table: Dict[str, Any]) -> Optional[str]:
    text_types = ("char", "text", "varchar", "nvarchar")
    for c in table.get("columns", []):
        dt = str(c.get("data_type", "")).lower()
        if any(tt in dt for tt in text_types):
            return c.get("name")
    return None


def detect_date_column(table: Dict[str, Any]) -> Optional[str]:
    return find_column_like(table, ["created_at", "createdon", "created_date", "updated_at", "date", "time"])


def detect_business_filters(table: Dict[str, Any]) -> List[str]:
    filters = []

    # soft delete patterns
    if col_exists(table, "is_deleted"):
        filters.append("is_deleted = false")
    if col_exists(table, "deleted"):
        filters.append("deleted = false")
    if col_exists(table, "deleted_at"):
        filters.append("deleted_at IS NULL")

    # active patterns
    if col_exists(table, "is_active"):
        filters.append("is_active = true")
    if col_exists(table, "active"):
        filters.append("active = true")

    # status patterns
    status_col = find_column_like(table, ["status", "state"])
    if status_col:
        filters.append(f"{status_col} = 'ACTIVE'")

    # multi-tenant patterns
    tenant_col = find_column_like(table, ["tenant_id", "org_id", "school_id", "company_id"])
    if tenant_col:
        filters.append(f"{tenant_col} = 1")

    return filters


# -----------------------------
# Prompt builder
# -----------------------------
def make_prompt(schema_text: str, question: str, sql: str) -> Dict[str, str]:
    return {
        "text": (
            "### Database Schema:\n"
            f"{schema_text}\n\n"
            "### Question:\n"
            f"{question}\n\n"
            "### SQL:\n"
            f"{sql}"
        )
    }


# -----------------------------
# FK graph
# -----------------------------
def build_fk_graph(tables: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    graph = defaultdict(list)
    for t in tables:
        tname = t.get("name")
        for fk in t.get("foreign_keys", []) or []:
            rt = fk.get("references_table")
            if tname and rt:
                graph[tname].append((rt, fk))
    return graph


def find_join_paths(graph: Dict[str, List[Tuple[str, Dict[str, Any]]]], start: str, max_depth: int = 3):
    paths = []
    q = deque()
    q.append((start, []))

    visited = set()

    while q:
        cur, path = q.popleft()

        if len(path) >= 1:
            key = (start, tuple(p[0] for p in path))
            if key not in visited:
                visited.add(key)
                paths.append(path)

        if len(path) == max_depth:
            continue

        for nxt, fk in graph.get(cur, []):
            q.append((nxt, path + [(nxt, fk)]))

    return paths


# -----------------------------
# SQL generation categories
# -----------------------------
def gen_basic_examples(db_type: str, schema_text: str, table: Dict[str, Any]) -> List[Dict[str, str]]:
    tname = table.get("name")
    tschema = table.get("schema", "")
    if not tname:
        return []

    full_t = normalize_table_ref(db_type, tschema, tname)

    id_col = find_id_column(table) or "id"
    text_col = find_text_column(table)
    num_col = find_numeric_column(table)
    date_col = detect_date_column(table)
    filters = detect_business_filters(table)

    examples = []
    base_where = ""
    if filters:
        base_where = " WHERE " + " AND ".join(filters[:2])

    examples.append(make_prompt(
        schema_text,
        f"Show latest 20 records from {tname} (apply business rules).",
        f"SELECT * FROM {full_t}{base_where} ORDER BY {id_col} DESC LIMIT 20;"
    ))

    examples.append(make_prompt(
        schema_text,
        f"Count total {tname} records (only active + not deleted if applicable).",
        f"SELECT COUNT(*) AS total_count FROM {full_t}{base_where};"
    ))

    if text_col:
        wc = " WHERE " + " AND ".join(filters[:2] + [f"{text_col} ILIKE '%test%'"]) if filters else f" WHERE {text_col} ILIKE '%test%'"
        examples.append(make_prompt(
            schema_text,
            f"Find {tname} rows where {text_col} contains 'test' with valid business filters.",
            f"SELECT * FROM {full_t}{wc} ORDER BY {id_col} DESC LIMIT 50;"
        ))

    if num_col:
        examples.append(make_prompt(
            schema_text,
            f"Calculate total {num_col} from {tname} using business filters.",
            f"SELECT SUM({num_col}) AS total_{num_col} FROM {full_t}{base_where};"
        ))

    if date_col:
        wc_parts = filters[:2] + [f"{date_col} >= NOW() - INTERVAL '30 days'"] if filters else [f"{date_col} >= NOW() - INTERVAL '30 days'"]
        wc = " WHERE " + " AND ".join(wc_parts)
        examples.append(make_prompt(
            schema_text,
            f"Show {tname} records for last 30 days with business filtering.",
            f"SELECT * FROM {full_t}{wc} ORDER BY {date_col} DESC LIMIT 50;"
        ))

    return examples


def gen_cte_analytics_examples(db_type: str, schema_text: str, table: Dict[str, Any]) -> List[Dict[str, str]]:
    tname = table.get("name")
    tschema = table.get("schema", "")
    if not tname:
        return []

    full_t = normalize_table_ref(db_type, tschema, tname)
    num_col = find_numeric_column(table)
    date_col = detect_date_column(table)
    filters = detect_business_filters(table)

    if not date_col:
        return []

    where_parts = filters[:2] + [f"{date_col} >= NOW() - INTERVAL '180 days'"]
    where_clause = " WHERE " + " AND ".join(where_parts)

    examples = []

    examples.append(make_prompt(
        schema_text,
        f"Generate month-wise count report for {tname} for last 6 months.",
        f"""
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', {date_col}) AS month,
        COUNT(*) AS total_count
    FROM {full_t}
    {where_clause}
    GROUP BY 1
)
SELECT *
FROM monthly
ORDER BY month;
""".strip()
    ))

    if num_col:
        examples.append(make_prompt(
            schema_text,
            f"Generate month-wise total {num_col} report for {tname} for last 6 months.",
            f"""
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', {date_col}) AS month,
        SUM({num_col}) AS total_{num_col}
    FROM {full_t}
    {where_clause}
    GROUP BY 1
)
SELECT *
FROM monthly
ORDER BY month;
""".strip()
        ))

    return examples


def gen_window_examples(db_type: str, schema_text: str, table: Dict[str, Any]) -> List[Dict[str, str]]:
    tname = table.get("name")
    tschema = table.get("schema", "")
    if not tname:
        return []

    full_t = normalize_table_ref(db_type, tschema, tname)
    num_col = find_numeric_column(table)
    date_col = detect_date_column(table)
    filters = detect_business_filters(table)

    if not num_col or not date_col:
        return []

    where_clause = ""
    if filters:
        where_clause = " WHERE " + " AND ".join(filters[:2])

    examples = []

    examples.append(make_prompt(
        schema_text,
        f"Rank top 20 rows in {tname} by {num_col} using window functions.",
        f"""
SELECT *
FROM (
    SELECT
        *,
        DENSE_RANK() OVER (ORDER BY {num_col} DESC) AS rnk
    FROM {full_t}
    {where_clause}
) ranked
WHERE rnk <= 20
ORDER BY {num_col} DESC;
""".strip()
    ))

    examples.append(make_prompt(
        schema_text,
        f"Generate running total of {num_col} by date for {tname}.",
        f"""
SELECT
    {date_col}::date AS day,
    SUM({num_col}) AS daily_total,
    SUM(SUM({num_col})) OVER (ORDER BY {date_col}::date) AS running_total
FROM {full_t}
{where_clause}
GROUP BY {date_col}::date
ORDER BY day;
""".strip()
    ))

    return examples


def gen_multijoin_examples(
    db_type: str,
    schema_text: str,
    base_table: Dict[str, Any],
    all_tables_index: Dict[str, Dict[str, Any]],
    graph: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    max_joins: int = 3
) -> List[Dict[str, str]]:
    base_name = base_table.get("name")
    base_schema = base_table.get("schema", "")
    if not base_name:
        return []

    base_ref = normalize_table_ref(db_type, base_schema, base_name)
    base_filters = detect_business_filters(base_table)

    join_paths = find_join_paths(graph, base_name, max_depth=max_joins)
    random.shuffle(join_paths)

    examples = []

    for path in join_paths[:3]:
        from_clause = f"FROM {base_ref} t1"
        where_filters = []

        if base_filters:
            where_filters.extend([f"t1.{f}" for f in base_filters[:2]])

        join_lines = []
        last_alias = "t1"
        last_schema = base_schema

        for idx, (nxt_table, fk) in enumerate(path, start=2):
            alias = f"t{idx}"

            left_col = fk.get("column")
            right_col = fk.get("references_column")
            ref_table = fk.get("references_table")

            if not left_col or not right_col or not ref_table:
                continue

            nxt_obj = all_tables_index.get(ref_table, {})
            nxt_schema = nxt_obj.get("schema", last_schema)
            nxt_ref = normalize_table_ref(db_type, nxt_schema, ref_table)

            join_lines.append(
                f"JOIN {nxt_ref} {alias} ON {last_alias}.{left_col} = {alias}.{right_col}"
            )

            last_alias = alias
            last_schema = nxt_schema

        where_clause = ""
        if where_filters:
            where_clause = "WHERE " + " AND ".join(where_filters)

        question = f"Join {base_name} with related tables (2-4 tables) and show first 50 rows using business rules."
        sql = (
            "SELECT t1.*\n"
            f"{from_clause}\n"
            + "\n".join(join_lines) + "\n"
            f"{where_clause}\n"
            "LIMIT 50;"
        )

        examples.append(make_prompt(schema_text, question, sql))

    return examples


# -----------------------------
# Main generator
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate domain SQL training data with live progress output")
    parser.add_argument("--schema-file", type=str, default=DEFAULT_SCHEMA_FILE)

    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", type=str, default=DEFAULT_VAL_FILE)

    parser.add_argument("--total-examples", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--schema-max-tables", type=int, default=12)
    parser.add_argument("--schema-max-cols", type=int, default=14)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-joins", type=int, default=3)

    args = parser.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.schema_file):
        raise FileNotFoundError(f"Schema file not found: {args.schema_file}")

    ensure_dir(args.out_dir)

    schema_json = read_json(args.schema_file)
    db_type = schema_json.get("database_type", "postgresql").lower()
    tables = schema_json.get("tables", [])

    if not tables:
        raise ValueError("No tables found in schema.json. Run schema_extractor.py first.")

    schema_text = schema_to_text(schema_json, max_tables=args.schema_max_tables, max_cols=args.schema_max_cols)
    all_tables_index = {t.get("name"): t for t in tables if t.get("name")}
    graph = build_fk_graph(tables)

    train_path = os.path.join(args.out_dir, args.train_file)
    val_path = os.path.join(args.out_dir, args.val_file)
    meta_path = os.path.join(args.out_dir, "dataset_meta.json")

    log("\n" + "=" * 110)
    log("✅ REAL-WORLD DOMAIN DATASET GENERATOR (LIVE)")
    log("=" * 110)
    log(f"Schema File      : {args.schema_file}")
    log(f"DB Type          : {db_type}")
    log(f"Tables Found     : {len(tables)}")
    log(f"Total Examples   : {args.total_examples}")
    log(f"Validation Ratio : {args.val_ratio}")
    log(f"Output Train     : {train_path}")
    log(f"Output Val       : {val_path}")
    log(f"Meta JSON        : {meta_path}")
    log("=" * 110 + "\n")

    # Clear old outputs first
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(val_path):
        os.remove(val_path)

    generated_unique = set()
    generated_rows: List[Dict[str, str]] = []

    # Live write buffer: we collect in memory first to split train/val properly
    # (If you want direct train/val streaming, tell me)
    total_target = args.total_examples

    # Generation loop with progress
    idx_table = 0
    while len(generated_rows) < total_target:
        for table in tables:
            if len(generated_rows) >= total_target:
                break

            idx_table += 1
            tname = table.get("name", "unknown_table")

            # Generate block of examples for current table
            batch = []
            batch.extend(gen_basic_examples(db_type, schema_text, table))
            batch.extend(gen_cte_analytics_examples(db_type, schema_text, table))
            batch.extend(gen_window_examples(db_type, schema_text, table))
            batch.extend(gen_multijoin_examples(
                db_type=db_type,
                schema_text=schema_text,
                base_table=table,
                all_tables_index=all_tables_index,
                graph=graph,
                max_joins=args.max_joins
            ))

            # Add unique items only
            added_now = 0
            for row in batch:
                if len(generated_rows) >= total_target:
                    break

                txt = row["text"]
                if txt not in generated_unique:
                    generated_unique.add(txt)
                    generated_rows.append(row)
                    added_now += 1

            # Print progress after each table
            log(
                f"{progress_bar(len(generated_rows), total_target)}  "
                f"Table: {tname}  | Added: +{added_now}"
            )

            # Stream-write "raw generation" file (optional debug)
            # If you want this, uncomment below:
            # debug_path = os.path.join(args.out_dir, "generation_live.jsonl")
            # with open(debug_path, "a", encoding="utf-8") as f:
            #     for r in batch:
            #         f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # safety: if no progress in many tables, break
            if idx_table > len(tables) * 3 and len(generated_rows) < 5:
                log("⚠ Not enough unique examples generated. Check FK relations and schema richness.")
                break

        # If we looped all tables but no more growth, stop
        if len(generated_rows) >= total_target:
            break

    # Trim
    generated_rows = generated_rows[:total_target]

    # Split train/val
    train_rows, val_rows = split_train_val(generated_rows, val_ratio=args.val_ratio)

    # Write final JSONL
    log("\n✅ Writing final JSONL files...")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Meta
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "schema_file": args.schema_file,
        "database_type": db_type,
        "tables_found": len(tables),
        "total_examples_requested": total_target,
        "examples_generated": len(generated_rows),
        "examples_train": len(train_rows),
        "examples_val": len(val_rows),
        "max_joins": args.max_joins,
        "schema_preview": schema_text[:2500]
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log("\n" + "=" * 110)
    log("✅ DATASET GENERATION COMPLETED SUCCESSFULLY")
    log("=" * 110)
    log(f"Train JSONL saved : {train_path}")
    log(f"Val JSONL saved   : {val_path}")
    log(f"Meta JSON saved   : {meta_path}")
    log(f"Total examples    : {len(generated_rows)}")
    log("=" * 110 + "\n")


if __name__ == "__main__":
    main()
