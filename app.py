from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import pymysql
import os
from flask import send_file
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from flask_restful import Resource, Api
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
import re
import base64
from PIL import Image
from io import BytesIO
import json
import pickle
import nltk
import random
import tensorflow as tf

#Implementasi Model
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer

app = Flask(__name__)
api = Api(app)
CORS(app)

# Database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'review',
}

# Folder for upload img
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'gif'}
@app.route('/get_image/<path:filename>')
def get_image(filename):
    return send_file(os.path.join(os.path.dirname(__file__), "uploads/{filename}"), mimetype='image/jpeg')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to insert data into MySQL
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Insert data into the table
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Function to insert article data into MySQL with image upload
def insert_article_to_mysql(article_data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_article as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS articles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    judul VARCHAR(255) NOT NULL,
                    gambar VARCHAR(255) NOT NULL,
                    link VARCHAR(255) NOT NULL
                )
            """
            cursor.execute(sql)

            # Save the uploaded image using secure_filename
            image_file = request.files['gambar']
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(image_file.filename)
                gambar = filename
                # image_file.save(os.path.join(app.config['UPLOAD_FOLDER'],image_file.filename))

                # Insert data into the table
                sql_insert = "INSERT INTO articles (judul, gambar, link) VALUES (%s, %s, %s)"
                cursor.execute(sql_insert, (article_data['judul'], gambar, article_data['link']))

                connection.commit()
                print("Article data with image inserted successfully!")
            else:
                print("Invalid or missing image file.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

def fetch_article_by_id(article_id):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            sql_select = "SELECT * FROM articles WHERE id = %s"
            cursor.execute(sql_select, (article_id,))
            result = cursor.fetchone()
            return result
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

def update_article_in_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Check if a new image file is provided
            if 'gambar' in data and data['gambar']:
                # Save the uploaded image using secure_filename
                image_file = data['gambar']
                if allowed_file(image_file.filename):
                    filename = secure_filename(image_file.filename)
                    gambar = image_file.filename
                    gambar1 = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                    image_file.save(gambar1)

                    # Update data in the table with the new image path
                    sql_update = "UPDATE articles SET judul = %s, link = %s, gambar = %s WHERE id = %s"
                    cursor.execute(sql_update, (data['judul'], data['link'], gambar, data['id']))
                else:
                    print("Invalid or missing image file.")
            else:
                # Update data in the table without changing the image
                sql_update = "UPDATE articles SET judul = %s, link = %s WHERE id = %s"
                cursor.execute(sql_update, (data['judul'], data['link'], data['id']))

            connection.commit()
            print("Article data updated successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Function to delete article from MySQL
def delete_article_from_mysql(article_id):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            sql_delete = "DELETE FROM articles WHERE id = %s"
            cursor.execute(sql_delete, (article_id,))
            connection.commit()
            print("Article deleted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Function to get all articles from MySQL
def get_all_articles():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            sql_select = "SELECT * FROM articles"
            cursor.execute(sql_select)
            result = cursor.fetchall()
            return result
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

def get_all_sentimen():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            sql_select = "SELECT * FROM hasil_model"
            cursor.execute(sql_select)
            result = cursor.fetchall()
            return result
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors_headers)

# Function to check admin login
def check_admin_login(username, password):
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    try:
        # Query to get admin data based on username
        query = 'SELECT * FROM admin WHERE username = %s'
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        print(result)

        if result:
            hashed_password = result[2]  # Ensure this index matches the table structure

            # Check the password match with the hashed password in the database
            if check_password_hash(hashed_password, password):
                return True, 'Login Success'
            else:
                return False, 'Invalid password'
        else:
            return False, 'User not found'
    except Exception as e:
        print(f"Error: {e}")
        return False, f"An error occurred during login: {e}"
    finally:
        cursor.close()
        connection.close()

def perform_login(username, password):
    # Lakukan verifikasi login sesuai kebutuhan proyek Anda
    # Misalnya, bandingkan dengan data yang tersimpan di database

    # Contoh sederhana: Jika username dan password sesuai, kembalikan sukses
    if username == 'user' and password == 'pass':
        return True, 'Login successful'
    else:
        return False, 'Invalid username or password'

# Route to render the form
@app.route('/')
def index():
    sentimen_data = get_all_sentimen()
    return render_template('index.html', data=sentimen_data)

@app.route('/admin')
def admin():
    return render_template('berita.html')

@app.route('/article')
def article():
    # Assume get_all_articles() returns a list of tuples representing articles
    articles_data = get_all_articles()
    return render_template('daftarArtikel.html', data=articles_data)

# Route to handle form submission
@app.route('/submit', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)

    #DBMS
    import pandas as pd
    import pymysql

    def is_table_empty(table, host='localhost', user='root', password='', database='review'):
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()
        
        # Check if the table is empty
        query_check_empty = f"SELECT COUNT(*) FROM {table}"
        cursor.execute(query_check_empty)
        count_result = cursor.fetchone()[0]

        # Close the cursor and the database connection
        cursor.close()
        connection.close()

        return count_result == 0

    #Implemnetasi Model dengan Data Baru
    def read_mysql_table(table, host='localhost', user='root', password='', database='review'):
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()
        
        query = f"SELECT * FROM {table}"
        cursor.execute(query)
        result = cursor.fetchall()
        
        # Convert the result to a Pandas DataFrame
        df = pd.DataFrame(result)
        
        # Assign column names based on the cursor description
        df.columns = [column[0] for column in cursor.description]
        
        # Close the cursor and the database connection
        cursor.close()
        connection.close()
        
        return df

    table_name = 'input_review'

    if not is_table_empty(table_name):
        df = read_mysql_table(table_name)
        # #menyimpan tweet. (tipe data series pandas)
        data_content = df['review']

        # casefolding
        data_casefolding = data_content.str.lower()
        data_casefolding.head()

        #filtering

        #url
        filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", str(tweet)) for tweet in data_casefolding]
        #cont
        filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
        #punctuatuion
        filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]
        #  hapus #tagger
        filtering_tagger = [re.sub(r'#([^\s]+)', '', tweet) for tweet in filtering_punctuation]
        #numeric
        filtering_numeric = [re.sub(r'\d+', ' ', tweet) for tweet in filtering_tagger]

        # # filtering RT , @ dan #
        # fungsi_clen_rt = lambda x: re.compile('\#').sub('', re.compile('rt @').sub('@', x, count=1).strip())
        # clean = [fungsi_clen_rt for tweet in filtering_numeric]

        data_filtering = pd.Series(filtering_numeric)

        # #tokenize
        tknzr = TweetTokenizer()
        data_tokenize = [tknzr.tokenize(tweet) for tweet in data_filtering]
        data_tokenize

        #slang word
        path_dataslang = open("Data/kamus kata baku-clear.csv")
        dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

        def replaceSlang(word):
            if word in list(dataslang[0]):
                indexslang = list(dataslang[0]).index(word)
                return dataslang[1][indexslang]
            else:
                return word

        data_formal = []
        for data in data_tokenize:
            data_clean = [replaceSlang(word) for word in data]
            data_formal.append(data_clean)
            len_data_formal = len(data_formal)
            # print(data_formal)
            # len_data_formal

        nltk.download('stopwords')
        default_stop_words = nltk.corpus.stopwords.words('indonesian')
        stopwords = set(default_stop_words)

        def removeStopWords(line, stopwords):
            words = []
            for word in line:  
                word=str(word)
                word = word.strip()
                if word not in stopwords and word != "" and word != "&":
                    words.append(word)

            return words
        reviews = [removeStopWords(line,stopwords) for line in data_formal]

        # Specify the file path of the pickle file
        file_path = 'model/cv.pickle'

        # Read the pickle file
        with open(file_path, 'rb') as file:
            data_train = pickle.load(file)
            
        # pembuatan vector kata
        vectorizer = TfidfVectorizer()
        train_vector = vectorizer.fit_transform(data_train)
        reviews2 = [" ".join(r) for r in reviews]

        load_model = pickle.load(open('model/svm_model.pkl','rb'))

        result = []

        for test in reviews2:
            test_data = [str(test)]
            test_vector = vectorizer.transform(test_data).toarray()
            pred = load_model.predict(test_vector)
            result.append(pred[0])
            
        unique_labels(result)

        df['label'] = result

        def delete_all_data_from_table(table, host='localhost', user='root', password='', database='review'):
            # Establish a connection to the MySQL database
            connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()
            
            # Delete all data from the specified table
            query = f"DELETE FROM {table}"
            cursor.execute(query)
            
            # Commit the changes
            connection.commit()
            
            # Close the cursor and the database connection
            cursor.close()
            connection.close()

        delete_all_data_from_table('input_review')

        def insert_df_into_hasil_model(df, host='localhost', user='root', password='', database='review'):
            # Establish a connection to the MySQL database
            connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )

            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            # Insert each row from the DataFrame into the 'hasil_model' table
            for index, row in df.iterrows():
                query = "INSERT INTO hasil_model (id_review, nama, tanggal, review, label) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(query, (row['id_review'], row['nama'], row['tanggal'], row['review'], row['label']))

            # Commit the changes
            connection.commit()

            # Close the cursor and the database connection
            cursor.close()
            connection.close()

        insert_df_into_hasil_model(df)

        table_name = 'hasil_model'
        hasil_df = read_mysql_table(table_name)
        hasil_df.to_csv('Data/hasil_model.csv')
        data = pd.read_csv('data/hasil_model.csv')
    else:
        # Membaca data dari file CSV
        data = pd.read_csv('data/hasil_model.csv')

    data = data[['review', 'label']]

    return jsonify({'status': 'success'})


# Route to handle article submission with image upload
@app.route('/submit_article', methods=['POST', 'OPTIONS'])
def submit_article_with_image():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    # Check if the request contains the expected form fields
    # if 'judul' not in request.form or 'gambar' not in request.files or 'link' not in request.form:
    #     return jsonify({'status': 'error', 'message': 'Invalid request data'})

    article_data_to_insert = {
        'judul': request.form['judul'],
        'gambar':request.files['gambar'].filename,  # Save only the filename
        'link': request.form['link']
    }

    insert_article_to_mysql(article_data_to_insert)

    # Save the uploaded image using secure_filename
    image_file = request.files['gambar']
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        gambar = image_file.filename
        gambar1 = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(gambar1)

    return redirect(url_for('article'))

# Route to handle article edit
@app.route('/edit_article/<int:article_id>', methods=['GET', 'POST'])
def edit_article(article_id):
    if request.method == 'GET':
        # Fetch the article data based on the article_id
        article_data = fetch_article_by_id(article_id)  # Implement this function as needed
        return render_template('edit_article.html', article_data=article_data)

    elif request.method == 'POST':
        # Process the form submission for article editing
        edited_data = {
            'id': article_id,
            'judul': request.form['judul'],
            'link': request.form['link'],
            'gambar': request.files['gambar']
        }
        update_article_in_mysql(edited_data)  # Implement this function to update the article
        return redirect(url_for('article'))  # Redirect to the article list page after editing

@app.route('/delete_article/<int:article_id>', methods=['GET', 'POST'])
def delete_article(article_id):
    # Implement the logic to delete the article by its ID
    delete_article_from_mysql(article_id)  # Implement this function to delete the article
    return redirect('/article')  # Redirect to the article list page after deletion

# Route to get all articles
@app.route('/get_articles', methods=['GET', 'OPTIONS'])
def get_articles():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    articles = get_all_articles()
    articles1 = [{'id': row[0], 'judul': row[1], 'gambar': row[2], 'link': row[3]} for row in articles]
    return jsonify({'status': 'success', 'articles': articles1})

# Route to login
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def admin_login():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    success, message = check_admin_login(username, password)
    return jsonify({'success': success, 'message': message})

# Endpoint untuk login mobile
@app.route('/api/login-mobile', methods=['POST'])
def login_mobile():
    try:
        data = request.get_json()
        username = data['username']
        password = data['password']

        success, message = perform_login(username, password)

        if success:
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'failure', 'message': message}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Model deteksi web
model_deteksi_sampah = load_model('model/model_web.h5')

def base64_to_pil(img_base64):
    # Mengonversi base64 ke bytes
    img_data = base64.b64decode(img_base64)

    # Membuka gambar dengan PIL
    pil_image = Image.open(BytesIO(img_data))
    pil_image = pil_image.resize((256, 256))  # Resize gambar ke ukuran yang diharapkan
    return pil_image

def model_predict(img, model):
    x = img_to_array(img)
    x = x.reshape(-1, x.shape[0], x.shape[1], 3)
    x = x.astype('float32')
    x = x / 255.0
    preds = model.predict(x)
    return preds

target_names = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
display_names = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
label_mapping = dict(zip(target_names, display_names))

# menyimpan histori sampah
deteksi_sampah_history = []

class Predict(Resource):
    def post(self):
        try:
            dataObject = request.json
            data = dataObject['message']

            # Tambahkan log sebelum pemrosesan gambar
            print("Received image data:", data)

            img = base64_to_pil(data)

            # Tambahkan log setelah pemrosesan gambar
            print("Image processed successfully")

            pred = model_predict(img, model_deteksi_sampah)
            hasil_label = label_mapping[target_names[np.argmax(pred)]]
            hasil_prob = "{:.2f}".format(100 * np.max(pred))

            # Masukan hasil deteksi ke histori
            deteksi_sampah_history.append({
                'nama' : hasil_label,
                'akurasi' : hasil_prob
            })

            return {'message': 'Berhasil melakukan prediksi', 'nama': hasil_label, 'akurasi': hasil_prob}, 200

        except Exception as e:
            print("EXCEPTION in Predict POST:", str(e))
            return {'message': str(e), 'nama': '', 'probability': ''}, 400

api.add_resource(Predict, '/api/deteksi-sampah', methods=['POST'])

# Model chatbot website
class Chatbot:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.intents = json.loads(open("./model/intents.json").read())
        self.words = pickle.load(open('./model/words.pkl', 'rb'))
        self.classes = pickle.load(open('./model/classes.pkl', 'rb'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence):
        # Gunakan self.lemmatizer
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, self.words, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text)
        res = self.getResponse(ints)
        return res
    
# Buat instance Chatbot
chatbot_instance = Chatbot('./model/chatbot_model.h5')

#========================================================================================

@app.route('/uploadFileAndroid', methods=['POST'])
def uploadFileAndroid():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'


    file.save('android/images/' + file.filename)

    return 'File uploaded successfully'

#deteksi mobile
@app.route('/receive_json', methods=['POST'])
def receive_json():
    try:
        dataObject = request.json
        filename = dataObject['message']
        path = "android/images/"

        # Baca file gambar
        with open(path + filename, "rb") as image_file:
            img_data = image_file.read()

        # Konversi data gambar ke dalam format base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Buat objek gambar dengan PIL
        img = base64_to_pil(img_base64)

        # Load model TensorFlow Lite (.tflite)
        interpreter = tf.lite.Interpreter(model_path="model/model_mobile.tflite")
        interpreter.allocate_tensors()

        # Masukkan gambar ke dalam model
        input_details = interpreter.get_input_details()
        input_data = np.expand_dims(img_to_array(img).astype('float32') / 255.0, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Jalankan inferensi
        interpreter.invoke()

        # Dapatkan output
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Proses output sesuai kebutuhan
        hasil_label = label_mapping[target_names[np.argmax(output_data)]]
        hasil_prob = "{:.2f}".format(100 * np.max(output_data))
        response_data = {'message': 'Berhasil melakukan prediksi', 'nama': hasil_label, 'akurasi': hasil_prob}

        return jsonify(response_data), 200

    except Exception as e:
        print("EXCEPTION in PredictTFLite POST:", str(e))
        return {'message': str(e), 'nama': '', 'probability': ''}, 400

class PredictTFLite(Resource):
    def post(self):
        try:
            dataObject = request.json
            data = dataObject['message']

            # Tambahkan log sebelum pemrosesan gambar
            print("Received image data:", data)

            img = base64_to_pil(data)

            # Tambahkan log setelah pemrosesan gambar
            print("Image processed successfully")

            # Load model TensorFlow Lite (.tflite)
            interpreter = tf.lite.Interpreter(model_path="model/model_mobile.tflite")
            interpreter.allocate_tensors()

            # Masukkan gambar ke dalam model
            input_details = interpreter.get_input_details()
            input_data = img_to_array(img).astype('float32') / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Jalankan inferensi
            interpreter.invoke()

            # Dapatkan output
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Proses output sesuai kebutuhan
            hasil_label = label_mapping[target_names[np.argmax(output_data)]]
            hasil_prob = "{:.2f}".format(100 * np.max(output_data))

            return {'message': 'Berhasil melakukan prediksi', 'nama': hasil_label, 'akurasi': hasil_prob}, 200

        except Exception as e:
            print("EXCEPTION in PredictTFLite POST:", str(e))
            return {'message': str(e), 'nama': '', 'probability': ''}, 400

api.add_resource(PredictTFLite, '/api/deteksi-mobile', methods=['POST'])

#chatbot mobile
class ChatbotMobile(Resource):
    def __init__(self, model_path, intents_path):
        self.model_path = model_path
        self.intents_path = intents_path  # Hanya menyimpan path file intents

        with open(self.intents_path) as f:
            self.intents = json.load(f)  # Membaca file intents saat diperlukan

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.words = pickle.load(open('./model/words.pkl', 'rb'))
        self.classes = pickle.load(open('./model/classes.pkl', 'rb'))

        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, self.words, show_details=False)

        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], np.array([p], dtype=np.float32))
        self.interpreter.invoke()

        output_details = self.interpreter.get_output_details()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        results = output_data[0]
        error = 0.25
        filtered_results = [(i, r) for i, r in enumerate(results) if r > error]
        filtered_results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in filtered_results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})

        return return_list

    def get_response(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text)
        res = self.get_response(ints)
        return res

    def post(self):
        data = request.get_json()
        user_message = data.get('message', '')
        response = self.chatbot_response(user_message)
        return {'message': response}

# Tambahkan path untuk file intents
chatbot_mobile_instance = ChatbotMobile(model_path='./model/model_chatbot.tflite', intents_path='./model/intents.json')

# Tambahkan resource ke dalam Api
api.add_resource(ChatbotMobile, '/api/chatbot-mobile', resource_class_kwargs={
    'model_path': chatbot_mobile_instance.model_path,
    'intents_path': chatbot_mobile_instance.intents_path
})

@app.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    try:
        data = request.get_json()
        user_text = data["message"]
        response = chatbot_instance.chatbot_response(user_text)
        return jsonify({"message": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)