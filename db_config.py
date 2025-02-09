import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        # Konfigurasi koneksi ke database
        connection = mysql.connector.connect(
            host="localhost",  # Ganti dengan host database Anda
            user="root",       # Ganti dengan username database Anda
            password="",       # Ganti dengan password database Anda
            database="production_db"  # Ganti dengan nama database Anda
        )

        if connection.is_connected():
            return connection

    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Jangan lupa untuk mengganti 'your_database_name' dengan nama database yang benar
