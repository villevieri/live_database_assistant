# Live Database Information Assistant — Milestone 2

## Run on Windows
1. Keep `app.py`, `analytics.db`, and `requirements.txt` in this folder.
2. Open Command Prompt in this folder.
3. Activate your existing virtual environment: `.venv\Scripts\activate`
   (If this is a new folder and `.venv` is absent, create it with `python -m venv .venv` first.)
4. Install dependencies: `pip install -r requirements.txt`
5. Make sure Ollama is running and `qwen3:4b` is installed (`ollama list`).
6. Run: `streamlit run app.py`

## What this milestone adds
- User types a question in natural language.
- Ollama/Qwen3 generates SQL from the actual database schema.
- SQLite is opened read-only and uses an authorizer to deny database modifications.
- If generation, validation, or execution fails, the agent sends the error back to the model and retries up to twice.
- The model summarizes the retrieved result and the app displays the SQL and retry log.

Actual tables: products, customers, orders, order_items.
