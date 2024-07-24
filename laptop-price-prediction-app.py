import streamlit as st
import pickle
import numpy as np

# Load the model and the data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Apply custom CSS for styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            font-family: 'Arial', sans-serif;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.9); /* Lighter box color */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stNumberInput {
            margin-bottom: 20px;
        }
        .stSelectbox select, .stNumberInput input {
            cursor: pointer;
        }
        h1, h2 {
            color: black;
            text-align: center;
        }
        .stError {
            color: #FF6347;
        }
    </style>
""", unsafe_allow_html=True)

# Set the title
st.markdown("<h1>Laptop Price Predictor</h1>", unsafe_allow_html=True)

# Create a container for the inputs
with st.container():
    st.markdown("<h2>Please select the following details:</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox('Brand', df['Company'].unique())
        type = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
        weight = st.number_input('Weight of the Laptop (in kg)', format="%.2f")
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
        ips = st.selectbox('IPS', ['No', 'Yes'])

    with col2:
        screen_size = st.number_input('Screen Size (in inches)', format="%.1f", min_value=1.0)
        resolution = st.selectbox('Screen Resolution', [
            '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
            '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])
        cpu = st.selectbox('CPU', df['Cpu brand'].unique())
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
        gpu = st.selectbox('GPU', df['Gpu brand'].unique())
        os = st.selectbox('OS', df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # Validation checks
    if weight == 0 or screen_size == 0:
        st.error("Please enter a valid weight and screen size.")
    else:
        try:
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == 'Yes':
                ips = 1
            else:
                ips = 0

            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, 12)

            predicted_price = int(np.exp(pipe.predict(query)[0]))
            st.success(f"The predicted price of this configuration is ₹{predicted_price:,}")

        except ZeroDivisionError:
            st.error("Screen size cannot be zero. Please enter a valid screen size.")

# Add a footer
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: rgba(255, 255, 255, 0.9); text-align: center; padding: 10px 0; border-top: 1px solid #ddd;">
        <p style="margin: 0;">Laptop Price Predictor &copy; 2024</p>
    </div>
""", unsafe_allow_html=True)
