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
st.caption("Local AI generates SQL, retrieves database evidence, and retries failed queries.")

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
- Prefer simple SELECT queries with straightforward JOIN, GROUP BY,
  ORDER BY, and LIMIT clauses. Avoid subqueries unless necessary.
- When a table has an alias, always use that alias to reference its columns.
- Never use a table's original name to reference its columns after aliasing it.

Revenue and sales:
- Total revenue for a product is SUM(order_items.quantity * order_items.unit_price).
- Total units sold is SUM(order_items.quantity).
- Calculate historical sales revenue using SUM(order_items.quantity * order_items.unit_price).
- Never substitute products.price for order_items.unit_price when
  calculating historical sales revenue.
- If asked for highest total revenue, sort by total_revenue DESC.
- If asked for both revenue and units sold, return both metrics with
  aliases total_revenue and total_units_sold.
- Do not confuse revenue, units sold, and number of orders.
- For revenue calculations, use the order_items table with alias oi.
- Calculate revenue as SUM(oi.quantity * oi.unit_price).
- Calculate units sold as SUM(oi.quantity).
- If order_items is aliased as oi, never use order_items.quantity
  or order_items.unit_price elsewhere in the query.
- Join products only when product names or other product fields are needed.
- Do not join orders or customers unless the question requires their data.

Sorting and prices:
- If the user asks about prices, include the product name and price.
- If the user asks for highest or lowest values, include the value
  being compared and sort in the correct direction.
- If the user asks for a ranking, return both the item name and its
  ranking value.
  - For questions asking for the highest-revenue product and units sold,
  return product name, total_revenue, and total_units_sold in the
  outer SELECT. Calculate revenue as SUM(oi.quantity * oi.unit_price),
  calculate units as SUM(oi.quantity), and order by total_revenue DESC.
- For questions about the cheapest or most expensive products,
  use products.price to sort products.
- "Lowest price" or "cheapest" means ORDER BY price ASC.
- "Highest price" or "most expensive" means ORDER BY price DESC.
- Do not join other tables unless the question requires information
  from those tables.
- The relationship between orders and products is through order_items:
  orders.id = order_items.order_id and
  products.id = order_items.product_id.
- Never join orders directly to products using product_id,
  because orders has no product_id column.
- Preserve the requested sorting order in the final answer.

Filters and table aliases:
- Only apply filters explicitly requested by the user.
- Do not add unrelated filters or joins.
- When a table has an alias, always use that alias to reference its columns.
- Never use a table's original name to reference its columns after aliasing it.
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
- Customer names may contain a first name followed by a space and a
  numeric suffix, such as 'Bob 4' or 'Jack 5'.
- When a question asks how many customers have a given first name,
  count all matching customer records whose names begin with that
  first name followed by a space.
- Generate the filter dynamically from the name in the user's question.
  Never hardcode specific names.
- Use a parameter-free SQL LIKE pattern such as
  name LIKE 'Bob %', replacing Bob with the requested name.
- If the user provides a complete customer name including its number,
  use an exact match instead.


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
            "num_predict": 256,
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


def get_clarification_options(question):
    """Return clarification options for recognized vague questions."""
    q = question.lower().strip().rstrip("?!.")

    vague_product_phrases = [
        "which products are doing best",
        "which product is doing best",
        "which products perform best",
        "which product performs best",
        "best performing products",
        "best performing product",
    ]

    if any(phrase in q for phrase in vague_product_phrases):
        return {
            "question": question,
            "prompt": "What would you like to compare?",
            "options": {
                "Highest total revenue":
                    "Which five products generated the highest total revenue?",
                "Most units sold":
                    "Which five products sold the most units?",
                "Lowest price":
                    "Which five products have the lowest prices?",
            },
        }
    vague_customer_phrases = [
        "which customers are the best",
        "who are the best customers",
        "which customers are most valuable",
        "best customers",
        "top customers",
    ]

    if any(phrase in q for phrase in vague_customer_phrases):
        return {
            "question": question,
            "prompt": "How would you like to rank customers?",
            "options": {
                "Highest spending":
                    "Which five customers have spent the most money on orders?",
                "Most orders":
                    "Which five customers have placed the most orders?",
                "Most recent signup":
                    "Which five customers signed up most recently?",
            },
        }
    
    vague_sales_phrases = [
        "tell me about sales",
        "tell me about the sales",
        "give me an overview of sales",
        "give me a sales overview",
        "how are sales doing",
        "sales overview",
    ]

    if any(phrase in q for phrase in vague_sales_phrases):
        return {
            "question": question,
            "prompt": "What would you like to know about sales?",
            "options": {
                "Total revenue":
                    "What is the total revenue from all orders?",
                "Sales by product":
                    "Which five products generated the highest total revenue?",
                "Sales over time":
                    "How has total revenue changed over time, grouped by month?",
            },
        }

    return None




def summarize_results(question, sql, dataframe):
    """Create a concise answer using the exact database results."""
    if dataframe.empty:
        return "I couldn't find any records matching your question."

    columns = dataframe.columns.tolist()
    question_lower = question.lower()

    # Highest-revenue product with units sold
    if (
        "product_name" in columns
        and "total_revenue" in columns
        and "total_units_sold" in columns
    ):
        row = dataframe.iloc[0]
        product = row["product_name"]
        revenue = row["total_revenue"]
        units = row["total_units_sold"]

        return (
            f"{product} generated the highest total revenue, "
            f"with €{float(revenue):,.2f} in sales from "
            f"{int(units)} units sold."
        )

    # Single numeric result, such as an average or count
    if len(columns) == 1 and len(dataframe) == 1:
        value = dataframe.iloc[0, 0]

        if "customer" in question_lower and (
            "how many" in question_lower or "count" in question_lower
        ):
            return f"There are {int(value)} customers matching your criteria."

        return f"The result is {value}."

    # Fast path: display straightforward list results without another LLM call
    simple_list_columns = {
        "name", "signup_date", "country", "category", "price",
        "product_name", "customer_name", "order_date",
        "total_amount", "total_revenue", "total_units_sold"
    }

    if columns and all(column in simple_list_columns for column in columns):
        display_df = dataframe.head(50).copy()

        # Expand country codes when a country column is present
        country_names = {
            "DE": "Germany",
            "FI": "Finland",
            "FR": "France",
            "SE": "Sweden",
            "UK": "United Kingdom",
            "US": "United States",
        }

        if "country" in display_df.columns:
            display_df["country"] = display_df["country"].map(
                lambda value: country_names.get(value, value)
            )

        # Format the results as a readable list
        lines = []
        for _, row in display_df.iterrows():
            parts = []
            for column in display_df.columns:
                value = row[column]
                if column in ("price", "total_amount", "total_revenue"):
                    try:
                        value = f"€{float(value):,.2f}"
                    except (ValueError, TypeError):
                        pass
                parts.append(f"{column.replace('_', ' ')}: {value}")
            lines.append("- " + "; ".join(parts))

        return f"Found {len(dataframe)} matching records:\n" + "\n".join(lines)

    # General fallback: use the model for other questions
    records = dataframe.head(50).to_dict(orient="records")
    prompt = f"""
- Display all monetary amounts in euros (€) with two decimal places.
  Do not convert or change the underlying numerical values.
- When displaying country codes, use full country names where the
  mapping is known: DE = Germany, FI = Finland, FR = France,
  SE = Sweden, UK = United Kingdom, US = United States.
- Answer the user's question using only the SQL results below.
- Be concise and natural. Do not invent or change numbers.
- Show only information relevant to the question.
- Preserve names exactly as provided.
- Preserve the exact row order from the SQL results.
- Never reorder, sort, or rank the returned records yourself.
- When listing results, present them in the same order as provided.

Question:
{question}

SQL results:
{json.dumps(records, ensure_ascii=False, default=str)}
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Explain database results accurately and naturally.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
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

clarification = get_clarification_options(question) if question.strip() else None

if clarification:
    st.info(clarification["prompt"])
    selected_option = st.radio(
        "Choose an interpretation",
        options=list(clarification["options"].keys()),
        key="clarification_choice",
    )
    run_button = st.button("Continue with selected option", type="primary")
else:
    selected_option = None
    run_button = st.button(
        "Ask the AI assistant",
        type="primary",
        disabled=not question.strip(),
    )

with st.expander("Database schema"):
    st.code(schema, language="text")


def classify_question(question, schema):
    """Check whether the question can be answered using the database."""
    print(f"Question: {question}")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Determine whether the user's question can be answered "
                    "using the provided database schema.\n"
                    "\n"
                    "Return IN_SCOPE for questions about information stored "
                    "in the database, including counting, filtering, comparing, "
                    "sorting, and aggregating customers, customer names, countries, "
                    "products, prices, orders, and sales.\n"
                    "\n"
                    "Examples of IN_SCOPE questions:\n"
                    "- How many customers do we have?\n"
                    "- How many customers named Bob do we have?\n"
                    "- Which country has the most customers?\n"
                    "- Which product generated the highest revenue?\n"
                    "\n"
                    "Return OUT_OF_SCOPE only when the question requires information "
                    "that cannot be answered from the database, such as current weather "
                    "or unrelated external facts.\n"
                    "\n"
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
    print(f"Scope classification result: {result}")
    print(f"Original question: {question}")

    if result.startswith("OUT_OF_SCOPE"):
        return "OUT_OF_SCOPE"

    return "IN_SCOPE"




def validate_business_rules(sql, question):
    """Validate the revenue formula for sales-related questions."""
    import re

    q = question.lower()
    s = sql.lower()

    revenue_question = any(
        term in q
        for term in (
            "revenue",
            "sales value",
            "total sales",
            "highest sales",
            "most sales",
        )
    )

    if not revenue_question:
        return

    # Detect quantity multiplied by price inside SUM().
    wrong_price_formula = re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?quantity\s*\*\s*(?:\w+\.)?price\s*\)",
        s,
    )

    if wrong_price_formula:
        # Check whether the query also contains the correct formula.
        correct_formula = re.search(
            r"\bsum\s*\(\s*(?:\w+\.)?quantity\s*\*\s*(?:\w+\.)?unit_price\s*\)",
            s,
        )

        if not correct_formula:
            raise ValueError(
                "Revenue must be calculated using quantity * unit_price "
                "from order_items, not quantity * price."
            )



if run_button:
    if clarification and selected_option:
        question = clarification["options"][selected_option]
    total_start = time.perf_counter()
    scope_start = time.perf_counter()
    
    with st.spinner("Checking your question..."):
        try:
            scope = classify_question(question, schema)
        except Exception as exc:
            scope = "IN_SCOPE"
            st.warning(
                f"Scope check unavailable; proceeding with the query. "
                f"Details: {exc}"
            )

    print(
        f"Scope classification took "
        f"{time.perf_counter() - scope_start:.2f} seconds"
    )

    if scope == "OUT_OF_SCOPE":
        st.info(
            "I can answer questions about products, prices, customers, "
            "countries, orders, and sales in the database. "
            "I don't have access to live weather information or other "
            "external data."
        )
        st.stop()
    else:
        with st.spinner("Generating SQL and querying the database..."):
            
            current_sql, last_error, result_df = None, None, None
            attempt_log = []

            for attempt in range(MAX_RETRIES + 1):
                attempt_sql = None

                try:
                    attempt_sql = generate_sql(
                        question,
                        schema,
                        previous_sql=current_sql if attempt > 0 else None,
                        error=last_error if attempt > 0 else None,
                    )
                    current_sql = attempt_sql

                    validate_business_rules(current_sql, question)
                    result_df = validate_and_run_sql(current_sql)

                    attempt_log.append({
                        "attempt": attempt + 1,
                        "sql": attempt_sql,
                        "status": "Succeeded",
                        "error": None,
                    })

                    last_error = None
                    break

                except Exception as exc:
                    last_error = str(exc)
                    attempt_log.append({
                        "attempt": attempt + 1,
                        "sql": attempt_sql or "(SQL generation failed)",
                        "status": "Failed",
                        "error": last_error,
                    })


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
                    answer_start = time.perf_counter()

                    with st.spinner("Generating natural-language answer..."):
                        answer = summarize_results(
                            question=question,
                            sql=current_sql,
                            dataframe=result_df,
                        )

                    print(f"Answer generation took {time.perf_counter() - answer_start:.2f} seconds")
                    print(f"Total response time: {time.perf_counter() - total_start:.2f} seconds")

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
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Source: analytics.db — results retrieved directly from SQLite."
            )

            with st.expander("SQL generated by the model"):
                st.code(current_sql, language="sql")

            
            with st.expander("Agent retry log"):
                for entry in attempt_log:
                    st.markdown(
                        f"**Attempt {entry['attempt']}: {entry['status']}**"
                    )

                    st.code(entry["sql"], language="sql")

                    if entry["error"]:
                        st.error(entry["error"])






st.divider()
st.caption("Local inference via Ollama. Only read-only queries are allowed; up to 200 rows are fetched.")
