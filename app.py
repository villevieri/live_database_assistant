import time
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import ollama

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "analytics.db"
MODEL_NAME = "qwen2.5:3b"
MAX_RETRIES = 2

st.set_page_config(page_title="Live Database Assistant", page_icon="🗃️", layout="wide")
st.title("🗃️ Live Database Information Assistant")
st.caption("Milestone 2 — local AI generates SQL, retrieves database evidence, and retries failed queries.")

def connect_db():
    """Open SQLite in read-only mode."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DB_PATH}. Keep analytics.db in the same folder as app.py."
        )
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)

def get_schema():
    """Read actual table and column names from analytics.db."""
    with connect_db() as conn:
        tables = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()
        schema_lines, table_names = [], []
        for (table,) in tables:
            table_names.append(table)
            columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            schema_lines.append(
                f"{table}: " + ", ".join(f"{col[1]} ({col[2]})" for col in columns)
            )
        return "\n".join(schema_lines), table_names

def get_row_counts(table_names):
    with connect_db() as conn:
        return {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in table_names
        }

def clean_sql(text):
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def generate_sql(question, schema, previous_sql=None, error=None):
    """Ask Ollama for a single read-only SQLite query."""
    correction = ""
    if previous_sql and error:
        correction = f"""
Your previous query failed.
Previous SQL:
{previous_sql}

Error:
{error}

Correct the query using the schema and error. Do not repeat the same mistake.
"""
    prompt = f"""/no_think
Generate one SQLite query for the user's question.

Actual database schema:
{schema}


Rules:
- Return SQL only, with no markdown fences or explanation.
- The query must start with SELECT or WITH.
- Use only tables and columns in the schema.
- Never modify data or schema. Do not use INSERT, UPDATE, DELETE, DROP,
  ALTER, CREATE, REPLACE, PRAGMA, ATTACH, or DETACH.
- Use LIMIT 10 for list questions unless the user requests another count.
- Do not invent columns.
- Always return the columns needed to answer the user's question.
- Return only columns relevant to the user's question.

Revenue and sales:
- Total revenue for a product is SUM(order_items.quantity * order_items.unit_price).
- Total units sold is SUM(order_items.quantity).
- Never substitute products.price for order_items.unit_price when
  calculating historical sales revenue.
- If asked for highest total revenue, sort by total_revenue DESC.
- If asked for both revenue and units sold, return both metrics with
  aliases total_revenue and total_units_sold.
- Do not confuse revenue, units sold, and number of orders.

Sorting and prices:
- If the user asks about prices, include the product name and price.
- If the user asks for highest or lowest values, include the value
  being compared and sort in the correct direction.
- If the user asks for a ranking, return both the item name and its
  ranking value.

Filters and table aliases:
- Only apply filters explicitly requested by the user.
- Do not add unrelated filters or joins.
- When using a table alias, reference that table's columns consistently
  through the alias throughout the query.
- Use only valid SQLite syntax.

Customers and countries:
- The customers.country column stores country codes, not full country names.
- When the user mentions a country by name, map it to the stored code
  only when filtering or grouping by that country.
- Country mappings: Finland = FI, France = FR, Sweden = SE,
  United States = US, United Kingdom = UK, Germany = DE.
- For a request to list customers, show their names by default.
- Include signup_date only when the user asks about registration,
  signup dates, or customer history.
- Include customer IDs only when requested or needed to distinguish
  customers with otherwise identical names.

User question:
{question}

{correction}
"""
    start_time = time.perf_counter()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate exactly one valid SQLite SELECT query. "
                    "Return SQL only, without explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0,
            "num_predict": 128,
        },
        keep_alive="10m",
    )

    elapsed = time.perf_counter() - start_time
    print(f"Ollama SQL generation took {elapsed:.2f} seconds")

    raw_content = response["message"].get("content", "").strip()
    thinking = response["message"].get("thinking", "")

    print("OLLAMA RAW RESPONSE:", repr(response["message"]))
    print("OLLAMA CONTENT:", repr(raw_content))
    print("OLLAMA THINKING:", repr(thinking))

    if not raw_content:
        raise ValueError(
            "Ollama returned an empty content field. "
            "Check the terminal for the thinking output."
        )

    return clean_sql(raw_content)

def validate_and_run_sql(sql):
    """Run one query. SQLite's authorizer denies write/schema-changing operations."""
    candidate = sql.strip()
    if not candidate:
        raise ValueError("The model returned an empty query.")
    if ";" in candidate.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")
    first_word = re.match(r"([A-Za-z]+)", candidate)
    if not first_word or first_word.group(1).upper() not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT queries (including read-only WITH queries) are allowed.")

    blocked = {
        sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX, sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX, sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER, sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    }

    with connect_db() as conn:
        def authorizer(action, arg1, arg2, database_name, trigger_name):
            return sqlite3.SQLITE_DENY if action in blocked else sqlite3.SQLITE_OK

        conn.set_authorizer(authorizer)
        cursor = conn.execute(candidate)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(200)
        return pd.DataFrame(rows, columns=columns)

def summarize_results(question, sql, dataframe):
    """Ask the model to describe retrieved rows without adding unsupported facts."""
    records = dataframe.head(50).to_dict(orient="records")
    prompt = f"""
Answer the user's question using only the SQL result below.
Do not invent numbers or facts. If there are no rows, say no matching records were found.
Be concise.
Show only information relevant to the user's question.
Do not mention technical column names such as signup_date unless relevant.
Do not include dates or other details that the user did not ask for.
Do not repeat database field names in the answer.
Preserve customer names exactly as provided in the results.
If the user asks to list customers, present a simple numbered list of names.
If the user asks for a count, state the count clearly.
- Never confuse revenue with units sold, order counts, or other metrics.
- Preserve the meaning and units of every result column.
- If a result has total_revenue and total_units_sold, report the
  revenue as a monetary amount and units_sold as a number of units.
- Do not describe a revenue value as units sold.
Question:
{question}

SQL:
{sql}

Result rows (up to 50):
{json.dumps(records, ensure_ascii=False, default=str)}

Rows displayed: {len(dataframe)}
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Explain database results accurately; do not invent facts."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
    )
    return response["message"]["content"].strip()

try:
    schema, table_names = get_schema()
    counts = get_row_counts(table_names)
except Exception as exc:
    st.error(f"Could not read analytics.db: {exc}")
    st.stop()

st.success(f"Connected to analytics.db · Local model: {MODEL_NAME}")
cols = st.columns(len(table_names))
for col, table in zip(cols, table_names):
    col.metric(table, f"{counts[table]:,} rows")

question = st.text_input(
    "Ask a question about your database",
    placeholder="For example: Which 5 products have the highest total sales value?",
)
run_button = st.button("Ask the AI assistant", type="primary", disabled=not question.strip())

with st.expander("Database schema"):
    st.code(schema, language="text")

def classify_question(question, schema):
    """Check whether the question can be answered using the database."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Determine whether a question can be answered using "
                    "the provided database schema. "
                    "Questions about products, prices, customers, countries, "
                    "orders, and sales are in scope if the schema supports them. "
                    "Questions about unrelated external information, such as "
                    "weather, are out of scope. "
                    "Return exactly IN_SCOPE or OUT_OF_SCOPE."
                ),
            },
            {
                "role": "user",
                "content": f"Database schema:\n{schema}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0, "num_predict": 10},
    )

    result = response["message"]["content"].strip().upper()

    if result.startswith("OUT_OF_SCOPE"):
        return "OUT_OF_SCOPE"

    return "IN_SCOPE"

if run_button:
    with st.spinner("Checking your question..."):
        try:
            scope = classify_question(question, schema)
        except Exception as exc:
            scope = "IN_SCOPE"
            st.warning(
                f"Scope check unavailable; proceeding with the query. "
                f"Details: {exc}"
            )

    if scope == "OUT_OF_SCOPE":
        st.info(
            "I can answer questions about products, prices, customers, "
            "countries, orders, and sales in the database. "
            "I don't have access to live weather information or other "
            "external data."
        )

    else:
        with st.spinner("Generating SQL and querying the database..."):
            current_sql, last_error, result_df = None, None, None
            attempt_log = []

            for attempt in range(MAX_RETRIES + 1):
                try:
                    current_sql = generate_sql(
                        question,
                        schema,
                        previous_sql=current_sql if attempt > 0 else None,
                        error=last_error if attempt > 0 else None,
                    )
                    result_df = validate_and_run_sql(current_sql)
                    attempt_log.append(
                        f"Attempt {attempt + 1}: query succeeded."
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    attempt_log.append(
                        f"Attempt {attempt + 1}: {last_error}"
                    )

        # Display the results after the retry loop.
        if result_df is None:
            st.error("I could not answer that question after retrying.")

            with st.expander("Troubleshooting details"):
                st.write(attempt_log)
                if current_sql:
                    st.code(current_sql, language="sql")

        else:
            st.subheader("Answer")

            if result_df.empty:
                st.write(
                    "I couldn't find any records matching your question "
                    "in the database. Try changing your search criteria."
                )

            else:
                try:
                    with st.spinner("Generating natural-language answer..."):
                        answer = summarize_results(
                            question=question,
                            sql=current_sql,
                            dataframe=result_df,
                        )

                    st.write(answer)

                except Exception as exc:
                    st.warning(
                        f"Could not generate a natural-language answer: {exc}"
                    )
                    st.write(
                        f"The database query returned "
                        f"**{len(result_df)} row(s)**. "
                        "Review the retrieved data below."
                    )

            st.subheader("Retrieved data")
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Source: analytics.db — results retrieved directly from SQLite."
            )

            with st.expander("SQL generated by the model"):
                st.code(current_sql, language="sql")

            with st.expander("Agent retry log"):
                st.write(attempt_log)





st.divider()
st.caption("Local inference via Ollama. Only read-only queries are allowed; up to 200 rows are fetched.")
