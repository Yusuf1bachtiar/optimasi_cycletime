import mysql.connector
import hashlib
import io
import pandas as pd
import csv
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler  # Untuk normalisasi data
from sklearn.preprocessing import LabelEncoder  # Untuk encoding label kategorikal
from datetime import datetime, timedelta
from flask_mysqldb import MySQL
from flask import Flask, make_response, render_template, request, redirect, url_for, session, flash
from db_config import get_db_connection
from flask import Flask
from werkzeug.security import generate_password_hash


app = Flask(__name__)
app.secret_key = "your_secret_key"

# Fungsi untuk koneksi ke database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Tidak ada password
        database="production_db"
    )

# Route Login
# Route Login
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if not username or not password:
            flash("Username and password are required", "error")
            return render_template("login.html")

        # Koneksi ke database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Menyimpan hanya username dalam session
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html")

# Route Dashboard
@app.route("/dashboard")
def dashboard():
    # Pastikan pengguna sudah login
    if "username" not in session:
        return redirect(url_for("login"))

    # Koneksi ke database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Menghitung jumlah data di tabel 'users'
    cursor.execute("SELECT COUNT(*) AS total_users FROM users")
    total_users = cursor.fetchone()["total_users"]

    # Menghitung jumlah data di tabel 'data'
    cursor.execute("SELECT COUNT(*) AS total_data FROM prediksi")
    total_data = cursor.fetchone()["total_data"]

    conn.close()

    # Mengirim data ke template dashboard.html
    return render_template("dashboard.html", username=session["username"], total_users=total_users, total_data=total_data)


@app.route('/data', methods=['GET', 'POST'])
def data():
    try:
        data = get_data_from_db()
        if request.method == 'POST':
            selected_ids = request.form.getlist('selected_data')
            action = request.form.get('action')

            if not selected_ids:
                message = "Silakan pilih data terlebih dahulu."
                return render_template('data.html', data=data, message=message)

            if action == "laporan":
                laporan_data = get_selected_prediksi_data(selected_ids)
                return render_template('laporan.html', data=laporan_data)

            elif action == "hapus":
                delete_selected_prediksi_data(selected_ids)
                message = "Data berhasil dihapus."
                data = get_data_from_db()  # Ambil ulang data setelah penghapusan
                return render_template('data.html', data=data, message=message)

        return render_template('data.html', data=data)
    except Exception as e:
        return render_template('data.html', message=f"Terjadi kesalahan: {str(e)}")

def delete_selected_prediksi_data(selected_ids):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = f"DELETE FROM prediksi WHERE id IN ({','.join([str(id) for id in selected_ids])})"
    cursor.execute(query)
    connection.commit()
    cursor.close()
    connection.close()



def get_data_from_db():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM prediksi")  # Query untuk mengambil semua data prediksi
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return data

    
def get_selected_prediksi_data(selected_ids):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    query = f"SELECT * FROM prediksi WHERE id IN ({','.join([str(id) for id in selected_ids])})"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return data


@app.route("/modul_prediksi")
def modul_prediksi():
    return render_template("modul_prediksi.html")

# Flask Routes
@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    if request.method == 'POST':
        if 'import_csv' in request.form:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    if df.empty:
                        return render_template('prediksi.html', message="File CSV kosong. Harap unggah file yang valid.")
                    
                    required_columns = ['station', 'jumlah_karyawan', 'jenis_mesin', 'rework', 'cycle_time', 'kategori']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        return render_template('prediksi.html', message=f"Kolom berikut hilang dalam file CSV: {', '.join(missing_columns)}")
                    
                    session['df'] = df.to_json()
                    return render_template('prediksi.html', data=df.to_html(classes='table table-bordered'))
                except Exception as e:
                    return render_template('prediksi.html', message=f"Terjadi kesalahan saat membaca file CSV: {str(e)}")
            else:
                return render_template('prediksi.html', message="Harap unggah file CSV yang valid.")
        
        if 'encode_data' in request.form:
            if 'df' in session:
                df = pd.read_json(session['df'])
                df = encode_data(df)
                session['df'] = df.to_json()
                return render_template('prediksi.html', data=df.to_html(classes='table table-bordered'))
            else:
                return render_template('prediksi.html', message="Silakan impor file CSV terlebih dahulu.")

        if 'train_model' in request.form:
            if 'df' in session:
                df = pd.read_json(session['df'])
                df = encode_data(df)
                model, accuracy = train_naive_bayes(df)
                joblib.dump(model, 'model_naive_bayes.pkl')
                return render_template('prediksi.html', message=f"Model telah dilatih dengan akurasi: {accuracy:.2f}")
            else:
                return render_template('prediksi.html', message="Silakan impor dan encode data terlebih dahulu.")
        
        if 'predict' in request.form:
            if 'df' in session:
                df = pd.read_json(session['df'])
                if os.path.exists('model_naive_bayes.pkl'):
                    model = joblib.load('model_naive_bayes.pkl')
                    X_test = df.drop(columns=['kategori'])
                    predictions = make_predictions(model, X_test)

                    # Pemetaan hasil prediksi numerik ke kategorikal
                    category_map = {0: "moderate", 1: "optimal", 2: "most optimal"}
                    categorized_predictions = [category_map[pred] for pred in predictions]

                    # Tambahkan nomor indeks ke prediksi untuk template
                    predictions_with_index = [(i + 1, pred) for i, pred in enumerate(categorized_predictions)]

                    total_cycle_time = sum(df['cycle_time'].tolist())
                    total_waktu_kerja = 480
                    jumlah_batch = total_waktu_kerja // total_cycle_time if total_cycle_time > 0 else 0
                    total_units = jumlah_batch

                    return render_template(
                        'prediksi.html',
                        predictions=predictions_with_index,
                        total_units=f"{int(total_units)} unit",
                        jumlah_batch=f"{int(jumlah_batch)} batch",
                        enumerate=enumerate  # Tambahkan enumerate ke context template
                    )
                else:
                    return render_template('prediksi.html', message="Model belum dilatih.")
            else:
                return render_template('prediksi.html', message="Silakan impor data terlebih dahulu.")
        
        if 'save_prediction' in request.form:
            if 'df' in session:
                try:
                    df = pd.read_json(session['df'])
                    if os.path.exists('model_naive_bayes.pkl'):
                        model = joblib.load('model_naive_bayes.pkl')
                    else:
                        return render_template('prediksi.html', message="Model belum dilatih.")
                    
                    predictions = make_predictions(model, df.drop(columns=['kategori']))
                    predictions_str = ', '.join(str(pred) for pred in predictions)

                    total_cycle_time = sum(df['cycle_time'].tolist())
                    total_waktu_kerja = 480
                    jumlah_batch = total_waktu_kerja // total_cycle_time if total_cycle_time > 0 else 0
                    total_units = jumlah_batch

                    # Gunakan fungsi get_db_connection untuk koneksi
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    query = """
                    INSERT INTO prediksi (tanggal, total_unit, jumlah_batch, kategori_prediksi)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(query, (datetime.now(), total_units, jumlah_batch, predictions_str))
                    conn.commit()
                    cursor.close()
                    conn.close()

                    return render_template('prediksi.html', message="Hasil prediksi berhasil disimpan.")
                except Exception as e:
                    return render_template('prediksi.html', message=f"Terjadi kesalahan: {str(e)}")
            else:
                return render_template('prediksi.html', message="Silakan impor data terlebih dahulu.")

    return render_template('prediksi.html')




@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input data dari request
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])

    # Melakukan prediksi
    prediction = model.predict(input_df)

    # Misalnya, jika kamu ingin memeriksa prediksi dan memastikan itu adalah nilai yang diinginkan
    if np.any(prediction == 1):  # Cek apakah ada prediksi dengan nilai 1
        return jsonify({'prediction': 'Class 1 detected'})

    return jsonify({'prediction': str(prediction[0])})

def encode_data(df):
    from sklearn.preprocessing import LabelEncoder
    
    for col in df.columns:
        if df[col].dtype == 'object':  # Jika tipe data adalah string/kategorikal
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])  # Encode nilai kategorikal
    
    return df

# Fungsi untuk melatih model Naive Bayes
def train_naive_bayes(df):
    X = df.drop(columns=['kategori'])
    y = df['kategori']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Fungsi untuk membuat prediksi
def make_predictions(model, X_test):
    return model.predict(X_test)


# Fungsi untuk membuat prediksi
def make_predictions(model, X_test):
    return model.predict(X_test)

@app.route('/assembly', methods=['GET', 'POST'])
def assembly():
    conn = get_db_connection()
    if conn is None:
        return "Error connecting to the database. Please check your configuration.", 500
    
    cursor = conn.cursor(dictionary=True)
    
    # If the request method is POST, handle form submission (Insert or Edit)
    if request.method == 'POST':
        assembly_id = request.form.get('id')  # Check for 'id' to identify an edit action
        
        # Data to be saved
        station = request.form['station']
        jumlah_karyawan = request.form['jumlah_karyawan']
        jenis_mesin = request.form['jenis_mesin']
        waktu_mulai = request.form['waktu_mulai']
        waktu_selesai = request.form['waktu_selesai']
        tanggal_produksi = request.form['tanggal_produksi']
        tipe_produk = request.form['tipe_produk']
        cycle_time = request.form['cycle_time']
        kategori = request.form['kategori']
        rework = request.form.get('rework', '0')  # Default to '0' if not provided

        # If editing, update the assembly record
        if assembly_id:
            cursor.execute("""
                UPDATE assembly
                SET station=%s, jumlah_karyawan=%s, jenis_mesin=%s, waktu_mulai=%s, waktu_selesai=%s,
                    tanggal_produksi=%s, tipe_produk=%s, cycle_time=%s, kategori=%s, rework=%s
                WHERE id=%s
            """, (station, jumlah_karyawan, jenis_mesin, waktu_mulai, waktu_selesai, tanggal_produksi, tipe_produk, cycle_time, kategori, rework, assembly_id))
            conn.commit()
        else:
            # Insert new assembly data
            cursor.execute("""
                INSERT INTO assembly (station, jumlah_karyawan, jenis_mesin, waktu_mulai, waktu_selesai, 
                                      tanggal_produksi, tipe_produk, cycle_time, kategori, rework)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (station, jumlah_karyawan, jenis_mesin, waktu_mulai, waktu_selesai, tanggal_produksi, tipe_produk, cycle_time, kategori, rework))
            conn.commit()

    # Fetch all assembly data and available product types
    cursor.execute("SELECT * FROM assembly ORDER BY id DESC")
    data_assembly = cursor.fetchall()

    cursor.execute("SELECT DISTINCT tipe_produk FROM produk")
    tipe_produk = [row['tipe_produk'] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    # Pass data to the template
    return render_template('assembly.html', data_assembly=data_assembly, tipe_produk=tipe_produk, editing=False)



@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_assembly(id):
    conn = get_db_connection()
    if conn is None:
        return "Error connecting to the database. Please check your configuration.", 500
    
    cursor = conn.cursor(dictionary=True)
    
    # Fetch the current data for the given ID
    cursor.execute("SELECT * FROM assembly WHERE id=%s", (id,))
    row = cursor.fetchone()

    if row is None:
        return "Record not found", 404
    
    if request.method == 'POST':
        # Get the form data
        station = request.form.get('station', '')
        jumlah_karyawan = request.form.get('jumlah_karyawan', '')
        jenis_mesin = request.form.get('jenis_mesin', '')
        waktu_mulai = request.form.get('waktu_mulai', '')
        waktu_selesai = request.form.get('waktu_selesai', '')
        tanggal_produksi = request.form.get('tanggal_produksi', '')
        tipe_produk = request.form.get('tipe_produk', '')
        cycle_time = request.form.get('cycle_time', '')
        kategori = request.form.get('kategori', '')
        rework = request.form.get('rework', '')  # Ambil nilai rework dari form

        # Update the record in the database
        cursor.execute("""
            UPDATE assembly
            SET station=%s, jumlah_karyawan=%s, jenis_mesin=%s, waktu_mulai=%s, waktu_selesai=%s,
                tanggal_produksi=%s, tipe_produk=%s, cycle_time=%s, kategori=%s, rework=%s
            WHERE id=%s
        """, (station, jumlah_karyawan, jenis_mesin, waktu_mulai, waktu_selesai, tanggal_produksi, tipe_produk, cycle_time, kategori, rework, id))
        conn.commit()
        cursor.close()
        conn.close()

        # Redirect back to the assembly page
        return redirect(url_for('assembly'))
    
    # Get product types for the dropdown list
    cursor.execute("SELECT DISTINCT tipe_produk FROM assembly")
    tipe_produk = [row['tipe_produk'] for row in cursor.fetchall()]  # Ensure tipe_produk is assigned

    cursor.close()
    conn.close()

    # Render the edit page with pre-filled data
    return render_template('edit_assembly.html', row=row, tipe_produk=tipe_produk)


@app.route('/auto_fill/<int:id>', methods=['GET'])
def auto_fill(id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil data berdasarkan ID
    cursor.execute("SELECT * FROM assembly WHERE id = %s", (id,))
    row = cursor.fetchone()

    if not row:
        return "Data not found", 404

    # Ambil waktu mulai dan selesai dari row
    waktu_mulai = row['waktu_mulai']
    waktu_selesai = row['waktu_selesai']

    if waktu_mulai and waktu_selesai:
        # Pastikan waktu_mulai dan waktu_selesai adalah timedelta atau string
        if isinstance(waktu_mulai, timedelta) and isinstance(waktu_selesai, timedelta):
            # Jika dalam bentuk timedelta, konversi menjadi total detik
            start_seconds = waktu_mulai.total_seconds()
            end_seconds = waktu_selesai.total_seconds()

            # Hitung durasi dalam menit
            duration = (end_seconds - start_seconds) // 60
        elif isinstance(waktu_mulai, str) and isinstance(waktu_selesai, str):
            # Jika dalam bentuk string, parse ke datetime dan hitung durasi
            fmt = "%H:%M:%S"
            waktu_mulai_obj = datetime.strptime(waktu_mulai, fmt)
            waktu_selesai_obj = datetime.strptime(waktu_selesai, fmt)

            # Hitung durasi dalam menit
            duration = (waktu_selesai_obj - waktu_mulai_obj).seconds // 60
        else:
            flash("Invalid time format in data", "danger")
            return redirect(url_for('assembly'))
    else:
        duration = None

    # Tentukan kategori berdasarkan cycle_time
    if duration is not None:
        if duration > 30:
            kategori = "Moderate"
        elif duration <= 20:
            kategori = "Most Optimal"
        else:
            kategori = "Optimal"
    else:
        kategori = None

    # Update data di database
    cursor.execute("""
        UPDATE assembly
        SET cycle_time = %s, kategori = %s
        WHERE id = %s
    """, (duration, kategori, id))
    conn.commit()

    cursor.close()
    conn.close()

    # Redirect kembali ke halaman assembly
    flash(f"Data for ID {id} updated: cycle_time = {duration}, kategori = {kategori}", "success")
    return redirect(url_for('assembly'))




@app.route('/assembly_export_csv', methods=['GET', 'POST'])
def export_assembly_csv():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Jika ada form input untuk tanggal mulai dan tanggal selesai
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    # Jika tanggal mulai dan tanggal selesai diberikan, filter data berdasarkan tanggal
    if start_date and end_date:
        cursor.execute("""
            SELECT station, jumlah_karyawan, jenis_mesin, tipe_produk, cycle_time, kategori, rework
            FROM assembly
            WHERE tanggal_produksi BETWEEN %s AND %s
        """, (start_date, end_date))
    else:
        # Ambil semua data jika tidak ada filter
        cursor.execute("""
            SELECT station, jumlah_karyawan, jenis_mesin, tipe_produk, cycle_time, kategori, rework
            FROM assembly
        """)
    
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Buat nama file dengan format assembly_<date>.csv
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"assembly_{date_str}.csv"

    # Gunakan io.StringIO untuk buffer CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Header CSV
    writer.writerow(["station", "jumlah_karyawan", "jenis_mesin", "tipe_produk", "cycle_time", "kategori", "rework"])

    # Isi data
    for row in data:
        writer.writerow([
            row['station'],
            row['jumlah_karyawan'],
            row['jenis_mesin'],
            row['tipe_produk'],
            row['cycle_time'],
            row['kategori'],
            row['rework']
        ])

    # Reset posisi buffer ke awal
    output.seek(0)

    # Buat respons Flask
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    response.headers['Content-Type'] = 'text/csv'

    return response

@app.route('/quality', methods=['GET', 'POST'])
def quality():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Cek data yang belum ada di quality
    cursor.execute("""
        SELECT a.* FROM assembly a
        LEFT JOIN quality q ON a.station = q.station_name AND a.tanggal_produksi = q.tanggal_produksi
        WHERE a.rework = 1 AND q.id IS NULL
    """)
    new_rework_data = cursor.fetchall()

    # Tambahkan hanya data yang belum ada
    for row in new_rework_data:
        cursor.execute("""
            INSERT INTO quality (station_name, jumlah_karyawan, tanggal_produksi, tipe_produk, rework, cycle_time)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row['station'], row['jumlah_karyawan'], row['tanggal_produksi'], row['tipe_produk'], row['rework'], None))
        conn.commit()

    # Ambil semua data quality
    cursor.execute("SELECT * FROM quality")
    quality_data = cursor.fetchall()

    conn.close()
    return render_template('quality.html', quality_data=quality_data)

@app.route('/update_cycle_time_quality', methods=['POST'])
def update_cycle_time_quality():
    data = request.json
    id = data['id']
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil waktu mulai, selesai, dan tanggal produksi dari database
    cursor.execute("SELECT waktu_mulai, waktu_selesai, tanggal_produksi FROM quality WHERE id = %s", (id,))
    result = cursor.fetchone()
    
    if result and result['waktu_mulai'] and result['waktu_selesai']:
        cycle_time = calculate_cycle_time(result['waktu_mulai'], result['waktu_selesai'], result['tanggal_produksi'])

        # Update cycle time di database
        cursor.execute("UPDATE quality SET cycle_time = %s WHERE id = %s", (cycle_time, id))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "cycle_time": cycle_time})
    
    conn.close()
    return jsonify({"success": False, "message": "Waktu mulai dan selesai harus diisi!"})

def calculate_cycle_time(start_time, end_time, production_date):
    if not start_time or not end_time:
        return 0
    
    # Gabungkan tanggal produksi dengan waktu mulai & selesai
    start = datetime.strptime(f"{production_date} {start_time}", "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(f"{production_date} {end_time}", "%Y-%m-%d %H:%M:%S")

    # Jika waktu selesai lebih kecil dari waktu mulai, berarti melewati tengah malam
    if end < start:
        end += timedelta(days=1)  # Tambahkan 1 hari ke waktu selesai

    delta = end - start
    return delta.seconds // 60  # Konversi ke menit



@app.route('/export_quality')
def export_quality():
    conn = get_db_connection()
    query = "SELECT * FROM quality"
    df = pd.read_sql(query, conn)
    conn.close()

    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=quality_data.csv"}
    )


@app.route('/laporan')
def laporan():
    try:
        prediksi_data = get_prediksi_data()  # Ambil data prediksi dari database

        print("Data dari database:", prediksi_data)

        # Hitung total unit dan jumlah batch berdasarkan data di tabel laporan
        total_units = sum(int(row['total_unit']) for row in prediksi_data if 'total_unit' in row)
        jumlah_batch = sum(int(row['jumlah_batch']) for row in prediksi_data if 'jumlah_batch' in row)

        return render_template('laporan.html', 
                               data=prediksi_data, 
                               total_units=total_units, 
                               jumlah_batch=jumlah_batch)
    except Exception as e:
        print("Error:", str(e))
        return render_template('laporan.html', message=f"Terjadi kesalahan: {str(e)}")



def get_prediksi_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil data dari tabel prediksi
    cursor.execute("SELECT tanggal, total_unit, jumlah_batch, kategori_prediksi FROM prediksi")
    data = cursor.fetchall()

    cursor.close()
    conn.close()
    
    return data



# Route untuk halaman registrasi
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # Simpan password langsung tanpa hash

        # Cek apakah username sudah terdaftar
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Username sudah terdaftar, silakan pilih username lain.")
            return redirect(url_for('register'))
        
        # Insert data pengguna baru ke database tanpa hashing password
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                       (username, password))
        conn.commit()
        cursor.close()
        conn.close()

        flash("Registrasi berhasil! Silakan login.")
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Route Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
    