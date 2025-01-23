import psycopg2

def setup_postgresql():
    """
    Sets up the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute("CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';")
        cursor.execute("CREATE DATABASE mydb;")
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;")
        print("PostgreSQL setup completed successfully.")
    except Exception as e:
        print(f"Error during PostgreSQL setup: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
