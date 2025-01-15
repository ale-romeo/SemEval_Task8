# SemEval 2025 - Task 8: DataBench, Question-Answering over Tabular Data

This repository contains a complete workflow for performing **Question-Answering (QA) over tabular data** using machine learning models and a relational database backend. The project demonstrates how to dynamically load datasets, generate SQL queries from natural language questions using a pre-trained language model, execute the queries on a PostgreSQL database, and evaluate the model's performance.

## Features
- **Dynamic Dataset Handling**: Automatically loads tabular datasets and stores them in a PostgreSQL database.
- **Natural SQL Query Generation**: Uses the pre-trained `chatdb/natural-sql-7b` model to generate SQL queries from natural language questions.
- **Query Execution**: Executes the generated SQL queries on a PostgreSQL database.
- **Result Post-Processing**: Post-processes query results to match expected output formats.
- **Evaluation**: Evaluates the model's performance by comparing generated results with ground truth answers.

## Workflow
1. **Clone the Repository**
2. **Install Required Libraries**
3. **Load QA Pairs**
4. **Set Up PostgreSQL**
5. **Generate SQL Queries**
6. **Execute Queries and Post-Process Results**
7. **Evaluate the Model's Performance**

## Setup Instructions

### Prerequisites
- Python 3.8+
- PostgreSQL installed and running

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ale-romeo/SemEval_Task8.git
   cd SemEval_Task8
   ```

2. Start PostgreSQL and create a new database:
   ```bash
   sudo service postgresql start
   psql -U postgres
   CREATE DATABASE mydb;
   ```

3. Set up the PostgreSQL user and grant privileges:
   ```sql
   CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';
   GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
   ```

### Running the Project

1. **Load datasets and process QA pairs**:
   Run the main script to load datasets dynamically, generate SQL queries, and store results.

2. **Save and Evaluate Results**:
   After processing QA pairs, save the results and evaluate the model's accuracy.

### Example Usage
```bash
python main.py
```

## File Structure
```
.
├── sql_manager.py         # Module for handling PostgreSQL operations
├── README.md              # Project documentation
└── evaluation.py          # Evaluation script for comparing results
```

## Model
The project uses the `chatdb/natural-sql-7b` model from Hugging Face for SQL query generation. The model is fine-tuned for converting natural language questions into SQL queries.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for providing the `chatdb/natural-sql-7b` model.
- [PostgreSQL](https://www.postgresql.org/) for the relational database backend.
- [Cardiff NLP](https://github.com/cardiffnlp) for the `databench` dataset.

