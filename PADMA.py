import base64
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import numpy as np
import cv2

# --- Custom CSS for Modern UI ---
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #f8fafc 100%);
    }
    .block-container {
        padding: 2rem;
        border-radius: 18px;
        background: white;
        box-shadow: 0 4px 20px rgba(31, 38, 135, 0.1);
    }
    .section-title {
        color: #2563eb;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56,189,248,0.2);
    }
    .stTextInput > div > input, .stNumberInput input {
        border-radius: 8px;
        border: 1.5px solid #cbd5e1;
        padding: 0.5rem 1rem;
        background: #f8fafc;
    }
    .stTextInput > div > input:focus {
        border-color: #38bdf8;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.1);
    }
    .stAlert {
        background: #fff6f6;
        border-left: 4px solid #e02424;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def process_image(image):
    image = image.convert("RGB").resize((224, 224))
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
    processed_img = cv2.merge((b, g, r))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    return image, Image.fromarray(processed_img)

def predict_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    try:
        response = requests.post("https://backendapi-ctgr.onrender.com/predict", json={"file": img_data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Prediction API Error: {e}")
        return None

def main():
    st.image("logo.png", use_container_width=True)
    st.title("Diabetic Foot Ulcer Monitoring and Severity Assessment - PADMA")
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:2rem;'>
            <span style='font-size:1.2rem; color:#64748b;'>
                Empowering diabetic foot care with AI-driven insights
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        # Patient Information
        st.markdown("<div class='section-title'>Patient Information</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name *")
            dob = st.date_input("Date of Birth *")
            age = st.number_input("Age *", min_value=1, max_value=120, step=1)
        with col2:
            email = st.text_input("Email *")
            gender = st.selectbox("Gender *", ["Select Gender", "Male", "Female", "Other"])
            contact = st.text_input("Contact Number *")

        # Medical History
        st.markdown("<div class='section-title'>Medical History</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            diabetes = st.radio("Do you have diabetes? *", ["Yes", "No"], index=None)
            if diabetes == "Yes":
                diabetes_years = st.number_input("Years with Diabetes *", min_value=0, step=1)
        with col2:
            high_bp = st.radio("Do you have high blood pressure? *", ["Yes", "No"], index=None)
            medications = st.text_area("Current Medications")

        # Example Image Upload (no foot assessment section)
        st.markdown("<div class='section-title'>Upload Example Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG) *", type=["jpg", "jpeg", "png"])

        # Submit Button
        if st.button("Analyze & Generate Report"):
            missing = []
            if not full_name: missing.append("Full Name")
            if not email: missing.append("Email")
            if not dob: missing.append("Date of Birth")
            if not age: missing.append("Age")
            if gender == "Select Gender": missing.append("Gender")
            if not contact: missing.append("Contact Number")
            if diabetes is None: missing.append("Diabetes Status")
            if diabetes == "Yes" and diabetes_years is None: missing.append("Years with Diabetes")
            if high_bp is None: missing.append("Blood Pressure Status")
            if not uploaded_file: missing.append("Image Upload")

            if missing:
                st.error(f"Please complete the following required fields: {', '.join(missing)}")
            else:
                with st.spinner("Processing image and generating prediction..."):
                    image = Image.open(uploaded_file)
                    _, processed_image = process_image(image)
                    st.image(processed_image, caption="Processed Image", use_container_width=True)
                    predictions = predict_image(processed_image)
                    if predictions:
                        if "probabilities" in predictions and "labels" in predictions:
                            st.success("Analysis Complete!")
                            st.subheader("Results:")
                            for label, prob in zip(predictions["labels"], predictions["probabilities"]):
                                st.markdown(
                                    f"<div style='padding:1rem; background:#f8fafc; border-radius:8px; margin:0.5rem 0;'>"
                                    f"<b>{label}</b>: {float(prob):.2f}%"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                        elif "error" in predictions:
                            st.error(f"Backend Error: {predictions['error']}")
                        else:
                            st.error("Unexpected response from backend.")

if __name__ == "__main__":
    main()