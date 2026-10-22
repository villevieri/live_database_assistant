
# Live Database Information Assistant

A local AI-powered application that allows users to ask natural-language questions about a SQLite database. The application uses Streamlit for its interface and Ollama with Qwen 2.5 (3B) to interpret questions and generate SQL queries.

## Features

- **Natural-language queries:** Ask questions about products, customers, orders, and sales.
- **Local AI model:** Uses `qwen2.5:3b` through Ollama.
- **Automatic SQL generation:** Converts questions into SQL using the actual database schema.
- **Query validation:** Validates generated SQL and applies business rules before execution.
- **Read-only database access:** Uses SQLite read-only mode and an authorizer to block database modifications.
- **Retry mechanism:** Sends SQL generation, validation, or execution errors back to the model for correction, with up to two retries.
- **Clarification options:** Recognizes selected vague questions about products, customers, and sales and offers predefined interpretations.
- **Result summaries:** Presents retrieved data in a concise, natural-language answer where appropriate.
- **Euro formatting:** Displays monetary amounts using the euro symbol and two decimal places.
- **Transparency:** Displays retrieved data, generated SQL, and the agent retry log.
- **Scope checking:** Identifies questions outside the database's intended scope.

## Technology Stack

- Python
- Streamlit
- SQLite
- Pandas
- Ollama
- Qwen 2.5 (3B)

## Database

The application uses `analytics.db`, which contains four tables:

| Table | Description |
|---|---|
| `products` | Product names, categories, and prices |
| `customers` | Customer names, countries, and signup dates |
| `orders` | Orders, customer relationships, dates, and total amounts |
| `order_items` | Products, quantities, and unit prices associated with orders |

The application uses the order-item quantities and unit prices to calculate historical sales revenue.

## Requirements

- Python 3.11 or newer
- Ollama installed and running
- The `qwen2.5:3b` model downloaded locally
- Python dependencies listed in `requirements.txt`

## Installation and Setup (Windows)

### 1. Install Ollama and the model

Install Ollama from:

https://ollama.com/download

Open Command Prompt and download the model:

```bash
ollama pull qwen2.5:3b
```

Verify that the model is available:

```bash
ollama list
```

### 2. Open the project folder

Open Command Prompt in the project directory containing `app.py`, `analytics.db`, and `requirements.txt`.

### 3. Create a virtual environment

If the project does not already have a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL to open in your browser.

## Example Questions

- How many orders are in the database?
- How many customers are from Finland?
- Which five customers have spent the most money on orders?
- Which product category generated the highest revenue?
- Which products are doing best?
- Which customers are the best?
- Tell me about sales.

For recognized ambiguous questions, the application offers clarification options before generating SQL.

## Safety and Limitations

- The application is designed for read-only analytical queries.
- Generated SQL must pass validation before execution.
- The retry mechanism does not guarantee that every question will produce a correct query.
- Clarification detection uses predefined Python rules and does not recognize every possible ambiguous question.
- Answer quality and response time depend on the local model and available hardware.
- Monetary values are formatted in euros; the application does not perform currency conversion.
- Questions requiring external information, such as live weather forecasts, are outside the intended database scope.

## Project Structure

```text
live_database_assistant_milestone2/
├── app.py
├── analytics.db
├── requirements.txt
├── README.md
└── .gitignore
```

The `.venv` directory is created locally and should not be committed to version control.

## License

This project was developed as an educational project for Showcase Challenge 2.
