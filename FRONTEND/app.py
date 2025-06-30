from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from joblib import load
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import joblib
from tensorflow.keras.applications import MobileNetV2 



app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='plant'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Password not matched!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])
        
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')


from sklearn.preprocessing import LabelEncoder

# Define class indices and reverse them to map predicted classes back to labels
class_indices = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3, 'Background_without_leaves': 4, 'Blueberry___healthy': 5,
    'Cherry___Powdery_mildew': 6, 'Cherry___healthy': 7,
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 8, 'Corn___Common_rust': 9,
    'Corn___Northern_Leaf_Blight': 10, 'Corn___healthy': 11, 'Grape___Black_rot': 12,
    'Grape___Esca_(Black_Measles)': 13, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14,
    'Grape___healthy': 15, 'Orange___Haunglongbing_(Citrus_greening)': 16,
    'Peach___Bacterial_spot': 17, 'Peach___healthy': 18, 'Pepper,_bell___Bacterial_spot': 19,
    'Pepper,_bell___healthy': 20, 'Potato___Early_blight': 21, 'Potato___Late_blight': 22,
    'Potato___healthy': 23, 'Raspberry___healthy': 24, 'Soybean___healthy': 25,
    'Squash___Powdery_mildew': 26, 'Strawberry___Leaf_scorch': 27, 'Strawberry___healthy': 28,
    'Tomato___Bacterial_spot': 29, 'Tomato___Early_blight': 30, 'Tomato___Late_blight': 31,
    'Tomato___Leaf_Mold': 32, 'Tomato___Septoria_leaf_spot': 33,
    'Tomato___Spider_mites Two-spotted_spider_mite': 34, 'Tomato___Target_Spot': 35,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 36, 'Tomato___Tomato_mosaic_virus': 37,
    'Tomato___healthy': 38
}

class_labels = {v: k for k, v in class_indices.items()}

# Load the saved SVM model and label encoder
svm_model = joblib.load('ML-models/svm_model.joblib')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(list(class_indices.keys()))  # Initialize label encoder

# Load the pre-trained MobileNet model (without top layers) for feature extraction
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Function to predict image class using SVM model with MobileNet feature extraction
def predict_image_class(img_path, model, label_encoder):
    img_height, img_width = 224, 224  # Use the dimensions from your training process
    img = load_img(img_path, target_size=(img_height, img_width))  # Load image
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image data

    # Extract features using MobileNet
    features = feature_extractor.predict(img_array)

    # Make a prediction with the SVM model
    prediction = model.predict(features)
    predicted_class = label_encoder.inverse_transform(prediction)

    return predicted_class[0]  # Return class name

# Flask route for displaying predictions
@app.route('/view_data', methods=['GET', 'POST'])
def view_data():
    if request.method == 'POST':
        # Get the uploaded image
        myfile = request.files['image']
        if myfile:
            fn = myfile.filename
            mypath = os.path.join('static/uploads/', fn)
            myfile.save(mypath)

            # Make the prediction using the SVM model
            predicted_class = predict_image_class(mypath, svm_model, label_encoder)
            # Assuming 'results' function gets detailed info about the prediction
            prediction, causes, remedies, organic, inorganic = results(predicted_class)

            # Render the results on the page
            return render_template('view_data.html', 
                                   prediction=predicted_class, 
                                   causes=causes, 
                                   remedies=remedies, 
                                   organic=organic, 
                                   inorganic=inorganic, 
                                   file_path=mypath)
    return render_template('view_data.html')



@app.route('/algorithm',methods=['GET','POST'])
def algorithm():
    global x_train, x_test, y_train, y_test,df
    msg = None
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        
        if algorithm == "Decision":
            model_name = "Decision Tree"
            accuracy = 43           
            
        elif algorithm == "Random":   
            model_name = "Random Forest"
            accuracy = 85
        
        elif algorithm == "SVM":
            model_name = "Support Vector Machine"
            accuracy = 93
        
        elif algorithm == "CNN":
            model_name = "CNN"
            accuracy = 10

        elif algorithm == "MobileNet":
            model_name = "MobileNet"
            accuracy = 74

        elif algorithm == "VGG16":
            model_name = "VGG16"
            accuracy = 89

        elif algorithm == "VGG19":
            model_name = "VGG19"
            accuracy = 88

        elif algorithm == "XGB":
            model_name = "Xtreme Gradient Boost"
            accuracy = 86
        
        msg = f"Accuracy of {model_name} is {accuracy}%"
    return render_template('algorithm.html', accuracy = msg)




# Load the pre-trained model
model_path = 'DL-models/MobileNetModel_best.h5'
best_model = tf.keras.models.load_model(model_path)

# Define class indices and labels
class_indices = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3, 'Background_without_leaves': 4, 'Blueberry___healthy': 5,
    'Cherry___Powdery_mildew': 6, 'Cherry___healthy': 7,
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 8, 'Corn___Common_rust': 9,
    'Corn___Northern_Leaf_Blight': 10, 'Corn___healthy': 11, 'Grape___Black_rot': 12,
    'Grape___Esca_(Black_Measles)': 13, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14,
    'Grape___healthy': 15, 'Orange___Haunglongbing_(Citrus_greening)': 16,
    'Peach___Bacterial_spot': 17, 'Peach___healthy': 18, 'Pepper,_bell___Bacterial_spot': 19,
    'Pepper,_bell___healthy': 20, 'Potato___Early_blight': 21, 'Potato___Late_blight': 22,
    'Potato___healthy': 23, 'Raspberry___healthy': 24, 'Soybean___healthy': 25,
    'Squash___Powdery_mildew': 26, 'Strawberry___Leaf_scorch': 27, 'Strawberry___healthy': 28,
    'Tomato___Bacterial_spot': 29, 'Tomato___Early_blight': 30, 'Tomato___Late_blight': 31,
    'Tomato___Leaf_Mold': 32, 'Tomato___Septoria_leaf_spot': 33,
    'Tomato___Spider_mites Two-spotted_spider_mite': 34, 'Tomato___Target_Spot': 35,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 36, 'Tomato___Tomato_mosaic_virus': 37,
    'Tomato___healthy': 38
}
class_labels = {v: k for k, v in class_indices.items()}

# Function to process image and make prediction
def make_prediction(image_path):
    img_height, img_width = 224, 224  # Ensure the dimensions match the model's input size
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    predictions = best_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Define a confidence threshold (e.g., 0.5)
    confidence_threshold = 0.5
    if confidence < confidence_threshold:
        return "Unknown Class", confidence
    else:
        return class_labels[predicted_class_index], confidence

# Route for prediction
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get the uploaded file
        myfile = request.files['image']
        if myfile:
            fn = myfile.filename
            mypath = os.path.join('static/uploads/', fn)
            myfile.save(mypath)

            # Make the prediction
            predicted_class, confidence = make_prediction(mypath)
            prediction, causes, remedies, organic, inorganic = results(predicted_class)

            # Render the prediction page with the results
            return render_template('prediction.html', 
                                   prediction=prediction, 
                                   confidence=round(confidence * 100, 2),
                                   causes=causes, 
                                   remedies=remedies, 
                                   organic=organic, 
                                   inorganic=inorganic, 
                                   file_path=mypath)
    return render_template('prediction.html')
 

@app.route('/graph')
def graph():
    return render_template('graph.html')


def results(predicted_label):
    causes = "None"
    remedies = "None"
    organic = "None"
    Inorganic = "None"
    if predicted_label == "Apple___Apple_scab":
        prediction = "Apple Apple scab"
        causes = "Fungus in wet conditions."
        remedies = "Apply fungicides, prune infected parts, maintain orchard sanitation."
        organic = "organic solutions, consider using neem oil or copper-based fungicides"
        Inorganic = " Inorganic solutions include chemical fungicides like captan or myclobutanil."

    elif predicted_label == "Apple___Black_rot":
        prediction = "Apple Black rot"
        causes = "Fungus through wounds."
        remedies = "Prompt pruning, fungicides during the season, proper orchard hygiene."
        organic = "Use neem oil spray regularly to prevent and manage black rot."
        Inorganic = "Apply copper-based fungicides as a preventive measure against black rot."

    elif predicted_label == "Apple___Cedar_apple_rust":
        prediction = "Apple Cedar apple rust"
        causes = "Fungus spread from junipers."
        remedies = "Apply fungicides, remove junipers, enhance air circulation."
        organic = "Pruning infected branches and applying neem oil."
        Inorganic = "Spraying affected trees with copper fungicides"

    elif predicted_label == "Apple___healthy":
        prediction = "Apple healthy"
        causes = "No disease."
        remedies = "Regular pruning, sanitation, vigilant disease monitoring."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Background_without_leaves":
        prediction = "Upload a relavent leaf image"


    elif predicted_label == "Blueberry___healthy":
        prediction = "Blueberry healthy"
        causes = "No disease."
        remedies = "Effective crop management, soil care, pest control."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Cherry___healthy":
        prediction = "Cherry healthy"
        causes = "No disease."
        remedies = "Proper orchard care, regular pruning, disease monitoring."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Cherry___Powdery_mildew":
        prediction = "Cherry___Powdery_mildew"
        causes = "Fungus in humid conditions."
        remedies = "Fungicides, improved ventilation, prompt pruning."
        organic = "Milk solution: Diluted milk (1 part milk to 9 parts water) sprayed on the affected plants can help suppress powdery mildew growth due to its antifungal properties."
        Inorganic = " Sulfur-based fungicides are effective against powdery mildew and can be applied to cherry trees according to label instructions."

    elif predicted_label == "Corn___Cercospora_leaf_spot Gray_leaf_spot":
        prediction = "Corn (maize) Cercospora leaf spot Gray leaf spot"
        causes = "Fungus (Cercospora spp.)."
        remedies = "Crop rotation, resistant varieties, fungicides."
        organic = "Utilize neem oil spray or copper-based fungicides"
        Inorganic = "Apply chemical fungicides containing chlorothalonil or azoxystrobin."

    elif predicted_label == "Corn___Common_rust":
        prediction = "Corn (maize) Common rust"
        causes = "Fungus (Puccinia spp.)."
        remedies = "Resistant varieties, fungicides, good field hygiene."
        organic = "Utilize neem oil or a garlic-chili pepper spray."
        Inorganic = "Apply copper-based fungicides"

    elif predicted_label == "Corn___healthy":
        prediction = "Corn (maize) healthy"
        causes = "No disease."
        remedies = "Crop rotation, proper field management."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Corn___Northern_Leaf_Blight":
        prediction = "Corn (maize) Northern Leaf Blight"
        causes = "Fungus (Exserohilum turcicum)."
        remedies = "Resistant varieties, preventive fungicides, sanitation."
        organic = "Crop rotation with non-host plants, intercropping with legumes, using compost or manure for soil enrichment, applying neem oil or garlic extract as natural fungicides."
        Inorganic = "Foliar application of copper-based fungicides, spraying with synthetic fungicides like chlorothalonil or mancozeb."

    elif predicted_label == "Grape___Black_rot":
        prediction = "Grape Black rot"
        causes = "Fungus in wet conditions."
        remedies = "Pruning, fungicides during the season, vineyard hygiene."
        organic = "Utilize a mixture of neem oil and baking soda sprayed on affected plants. Regularly prune to enhance air circulation and sunlight exposure, minimizing damp conditions favorable for the disease."
        Inorganic = " Apply copper-based fungicides according to label instructions to control the spread of the disease."

    elif predicted_label == "Grape___Esca_(Black_Measles)":
        prediction = "Grape Esca (Black Measles)"
        causes = "Fungus through wounds."
        remedies = "Pruning infected vines, systemic fungicides."
        organic = "Implementing cultural practices like proper pruning, improving soil health with compost, and using organic fungicides like neem oil."
        Inorganic = "Applying copper-based fungicides for control."

    elif predicted_label == "Grape___healthy":
        prediction = "Grape healthy"
        causes = "No disease."
        remedies = "Regular pruning, vigilant monitoring, disease-resistant varieties."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        prediction = "Grape Leaf blight (Isariopsis Leaf Spot)"
        causes = "Fungus (Isariopsis spp.)."
        remedies = "Pruning, fungicides, vineyard sanitation."
        organic = "Regular application of neem oil or copper-based fungicides."
        Inorganic = "Fungicides containing chemicals like chlorothalonil or mancozeb."

    elif predicted_label == "Orange___Haunglongbing_(Citrus_greening)":
        prediction = "Orange Haunglongbing (Citrus greening)"
        causes = "Bacteria through insect vectors."
        remedies = "Remove infected trees, apply antibiotics, control vectors."
        organic = "Emphasize soil health through composting, utilizing neem oil, and employing biocontrol agents like beneficial nematodes."
        Inorganic = " Implementing copper-based fungicides and bactericides for disease management."

    elif predicted_label == "Peach___Bacterial_spot":
        prediction = "Peach Bacterial spot"
        causes = "Bacteria (Xanthomonas spp.)."
        remedies = "Prune, copper-based sprays, orchard sanitation."
        organic = "Copper-based fungicides, neem oil, and biocontrol agents."
        Inorganic = "Copper sulfate-based sprays."

    elif predicted_label == "Peach___healthy":
        prediction = "Peach healthy"
        causes = "No disease."
        remedies = "Proper orchard management, pruning, sanitation."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Pepper,_bell___Bacterial_spot":
        prediction = "Pepper, bell Bacterial spot"
        causes = "Bacteria (Xanthomonas spp.)."
        remedies = "Resistant varieties, copper-based sprays, field hygiene."
        organic = "mplement crop rotation, use resistant varieties, apply neem oil or copper-based fungicides."
        Inorganic = "Apply copper-based bactericides or chemical fungicides approved for organic production."

    elif predicted_label == "Pepper,_bell___healthy":
        prediction = "Pepper, bell healthy"
        causes = "No disease."
        remedies = "Proper crop care, irrigation, pest control, resistant varieties."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Potato___Early_blight":
        prediction = "Potato Early blight"
        causes = "Fungus (Alternaria solani)."
        remedies = "Crop rotation, fungicides, proper field hygiene."
        organic = "Implement crop rotation, use neem oil as a natural fungicide, apply compost tea as a foliar spray."
        Inorganic = "Apply copper-based fungicides such as Bordeaux mixture or copper hydroxide formulations."

    elif predicted_label == "Potato___healthy":
        prediction = "Potato healthy"
        causes = "No disease."
        remedies = "Crop rotation, proper field management."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Potato___Late_blight":
        prediction = "Potato Late blight"
        causes = "Oomycete (Phytophthora infestans)."
        remedies = "Fungicides, avoid wet conditions, crop rotation."
        organic = "Use copper-based fungicides or neem oil"
        Inorganic = "Apply chemical fungicides containing chlorothalonil or mancozeb"

    elif predicted_label == "Raspberry___healthy":
        prediction = "Raspberry healthy"
        causes = "No disease."
        remedies = "Proper care, pruning, pest control."
        organic = " "
        Inorganic = " "


    elif predicted_label == "Soybean___healthy":
        prediction = "Soybean healthy"
        causes = "No disease."
        remedies = "Crop rotation, proper field management."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Squash___Powdery_mildew":
        prediction = "Squash Powdery mildew"
        causes = "Fungus (Podosphaera spp.)."
        remedies = "Fungicides, proper spacing, remove infected plants."
        organic = "Neem oil spray or milk solution (diluted with water)"
        Inorganic = "Potassium bicarbonate spray"

    elif predicted_label == "Strawberry___healthy":
        prediction = "Strawberry healthy"
        causes = "No disease."
        remedies = "Proper care, disease-resistant varieties, pest control."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Strawberry___Leaf_scorch":
        prediction = "Strawberry Leaf scorch"
        causes = "Fungus (Diplocarpon earlianum)."
        remedies = "Fungicides, prune infected leaves, proper irrigation."
        organic = "Applying neem oil or a solution of garlic and water can help manage strawberry leaf scorch."
        Inorganic = "Using copper fungicides can be effective in controlling strawberry leaf scorch."

    elif predicted_label == "Tomato___Bacterial_spot":
        prediction = "Tomato Bacterial spot"
        causes = "Bacteria (Xanthomonas spp.)."
        remedies = "Resistant varieties, copper-based sprays, crop rotation."
        organic = "Neem oil spray mixed with garlic extract"
        Inorganic = "Copper-based fungicides."

    elif predicted_label == "Tomato___Early_blight":
        prediction = "Tomato Early blight"
        causes = "Fungus (Alternaria solani)."
        remedies = "Fungicides, proper spacing, remove infected leaves."
        organic = "Apply neem oil or potassium bicarbonate spray regularly to control fungal growth"
        Inorganic = "Use copper-based fungicides according to label instructions for effective control."

    elif predicted_label == "Tomato___healthy":
        prediction = "Tomato healthy"
        causes = "No disease."
        remedies = "Crop rotation, proper care, disease-resistant varieties."
        organic = " "
        Inorganic = " "

    elif predicted_label == "Tomato___Late_blight":
        prediction = "Tomato Late blight"
        causes = "Oomycete (Phytophthora infestans)."
        remedies = "Fungicides, avoid wet conditions, proper ventilation."
        organic = "Use copper-based fungicides or apply a solution of neem oil to control Tomato Late Blight."
        Inorganic = "Apply synthetic fungicides such as chlorothalonil or mancozeb to manage Tomato Late Blight."
        
    elif predicted_label == "Tomato___Leaf_Mold":
        prediction = "Tomato Leaf Mold"
        causes = "Fungus (Fulvia fulva)."
        remedies = "Fungicides, proper ventilation, remove infected leaves."
        organic = "Use a mixture of neem oil and water sprayed on the affected plants"
        Inorganic = "Apply copper-based fungicides according to label instructions."

    elif predicted_label == "Tomato___Septoria_leaf_spot":
        prediction = "Tomato Septoria leaf spot"
        causes = "Fungus (Septoria lycopersici)."
        remedies = "Fungicides, prune infected leaves, proper irrigation."
        organic = "Apply neem oil or copper fungicide."
        Inorganic = "Apply chemical fungicides such as chlorothalonil or mancozeb."

    elif predicted_label == "Tomato___Spider_mites Two-spotted_spider_mite":
        prediction = "Tomato Spider mites Two-spotted spider mite"
        causes = "Two-spotted spider mite (Tetranychus urticae)."
        remedies = "Predatory mites, insecticidal soaps, proper humidity control."
        organic = "Introduce beneficial insects like ladybugs or lacewings to prey on spider mites."
        Inorganic = "Apply a neem oil or insecticidal soap spray to control spider mite infestation."

    elif predicted_label == "Tomato___Target_Spot":
        prediction = "Tomato Target Spot"
        causes = "Fungus (Corynespora cassiicola)."
        remedies = "Fungicides, prune infected leaves, proper field hygiene."
        organic = "Utilize neem oil or copper-based fungicides."
        Inorganic = "Apply chemical fungicides like chlorothalonil or mancozeb."

    elif predicted_label == "Tomato___Tomato_mosaic_virus":
        prediction = "Tomato Tomato mosaic virus"
        causes = "Virus (Tomato mosaic virus)."
        remedies = "Remove infected plants, control aphids, use virus-free seeds."
        organic = "Implement crop rotation, use resistant tomato varieties, employ neem oil or garlic spray as natural repellents."
        Inorganic = "Apply copper-based fungicides or chemical sprays like mancozeb for control."

    elif predicted_label == "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        prediction = "Tomato Tomato Yellow Leaf Curl Virus"
        causes = "Virus (Tomato yellow leaf curl virus)."
        remedies = "Use resistant varieties, control whiteflies, remove infected plants."
        organic = "Implement crop rotation, use resistant/tolerant tomato varieties, employ reflective mulches, encourage beneficial insects, and apply neem oil or garlic spray."
        Inorganic = "Utilize chemical pesticides such as imidacloprid or spinosad, following recommended application rates and safety precautions"
        
    else:
        prediction = None
        causes = "Class not recognized."
        remedies = "Please provide causes and remedies for this class."
        organic = " Please provide organic solution for this class"
        Inorganic = "Please provide Inorganic for this class."

    return prediction, causes, remedies, organic, Inorganic



if __name__ == '__main__':
    app.run(debug = True)