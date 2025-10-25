import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# import model h5 yang sudah dibuat sebelumnya
try:
    model = tf.keras.models.load_model('my_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# import nutrition data
nutrition_data = pd.read_csv('fruitsnutrition.csv')
nutrition_data.columns = nutrition_data.columns.str.strip() #menghapus spasi di awal atau akhir kolom, bisa untuk menghindari kelebihan spasi

# memproses gambar sebelum dimasukkan ke model
def preprocess_image(image):
    # gambar dengan warna RGBA akan diubah ke RGB
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # mengubah ukuran dan normalisasi piksel ke [0,1]
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    # menambah dimensi untuk pemrosesan batch
    return np.expand_dims(image, axis=0)

# untuk memprediksi jeis buah dari gambar
def predict_fruit(image):
    # daftar kelas buah yang dapat diprediksi
    fruit_classes = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberry', 'watermelon']
    
    # memproses gambar kemudian diprediksi dengan model
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    
    # mengembalikan nama buah yang telah diprediksi
    return fruit_classes[predicted_class_index]

# untuk mendapatkan informasi nutrisi buah
def get_nutritional_info(fruit_name):
    # menyamakan kapitalisasi nama buah
    fruit_name = fruit_name.lower().capitalize()
    
    # mencari buah dalam data
    nutrition = nutrition_data[nutrition_data['fruit'].str.contains(fruit_name, case=False, na=False)]
    
    # mengembalikan informasi nutrisi yang telah ditemukan
    return nutrition.iloc[0] if not nutrition.empty else None

# untuk menampilkan informasi nutrisi
def display_nutritional_info(nutrition):
    # data ditampilkan dalam format tabel
    data={
        "Nutrient":["Energy","Protein","Total Fat","Carbohydrates","Calcium","Iron","Vit A","Vit C","Vit E"],
        "Amount":[
            f"{nutrition['energy (kcal/kJ)']} kcal",
            f"{nutrition['protein (g)']} g",
            f"{nutrition['total fat (g)']} g",
            f"{nutrition['carbohydrates (g)']} g",
            f"{nutrition['calcium (mg)']} mg",
            f"{nutrition['iron (mg)']} mg",
            f"{nutrition['vitamin A (IU)']} IU",
            f"{nutrition['vitamin C (mg)']} mg",
            f"{nutrition['vitamin E (mg)']} mg"
        ]
    }
    
    # membuat frame dan menampilkan informasil nutrisi
    df=pd.DataFrame(data)
    st.markdown(f"""
                <h5>Nutritional Information:</h5>
                """, unsafe_allow_html=True)
    st.dataframe(df,hide_index=True,use_container_width=True)

def main():
    # mengatur latar belakang dan desain halaman
    page_bg_img='''
    <style>
    header {visibility: hidden;}
    
    [data-testid="stAppViewContainer"]{
    background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)),url("https://images.pexels.com/photos/1128678/pexels-photo-1128678.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-size: cover;
    background-repeat: no-repeat;
    image-rendering: crisp-edges;
    }

    [data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
    }

    [data-testid="stToolbar"]{
    right: 2rem;
    }

    .blur-box {
        background: rgba(255, 255, 255, 0.8); 
        border-radius: 10px;
        padding: 20px; 
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
        backdrop-filter: blur(10px); 
        -webkit-backdrop-filter: blur(10px); 
    }

    .blur-box h1, .blur-box h2, .blur-box p {
        margin: 0;
        color: #000; 
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="blur-box">
            <h1>Fruity Nutrition</h1>
            <h3>Welcome to Fruity Nutrition!</h3>
            <p>
                What is Fruity Nutrition? Fruity Nutrition is a web-based application designed to help you identify 
                fruits and understand their nutritional values. By using your camera or uploading an image, 
                the app predicts the type of fruit and provides detailed information about its nutritional content.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Image Upload Section
    option=st.radio("Choose an option to scan Fruit:",("Upload File","Use Camera (Desktop)"))
    if option=="Upload File":
        # mengunggah file gambar
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # menampilkan gambar yang diunggah dan mengubah format gambar menjadi BGR
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # membuat tombol unruk melakukan prediksi
            if st.button("Predict Fruit"):
                with st.spinner("Predicting..."):
                    predicted_fruit = predict_fruit(image_cv).capitalize()
                    st.markdown(f"""
                                <h5>Predicted Fruit:</h5>
                                <div class="blur-box">
                                    <span style="color: black; font-size: 20px; font-weight: bold;">{predicted_fruit}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    # menampilkan informasi nutrisi buah yang diprediksi
                    nutrition = get_nutritional_info(predicted_fruit)
                    if nutrition is not None:
                        display_nutritional_info(nutrition)
                    else:
                        st.write("Nutritional information not available for this fruit.")
    # Webcam Capture Section
    elif option=="Use Camera (Desktop)":
        st.write("Webcam Capture")
        # Checkbox untuk mengakifkan kamera
        enable_camera = st.checkbox("Enable Camera")
        if enable_camera:
            # menggunakan input kamera untuk mengambil foto
            camera_photo = st.camera_input("Capture a photo")
            if camera_photo is not None:
                # menampilkan gambar yang diambil
                image = Image.open(camera_photo)
                st.image(image, caption="Captured Image", use_container_width=True)
                # mengkonversi gambar jadi format BGR
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                # tombol untuk memprediksi jenis buah yang telah diambil
                if st.button("Predict Fruit"):
                    with st.spinner("Predicting..."):
                        predicted_fruit = predict_fruit(image_cv).capitalize()
                        st.markdown(f"""
                                <h5>Predicted Fruit:</h5>
                                <div class="blur-box">
                                    <span style="color: black; font-size: 20px; font-weight: bold;">{predicted_fruit}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        # menampilkan informasi nutrisi buah yang diprediksi
                        nutrition = get_nutritional_info(predicted_fruit)
                        if nutrition is not None:
                            display_nutritional_info(nutrition)
                        else:
                            st.write("Nutritional information not available for this fruit.")
if __name__ == "__main__":
    main()
