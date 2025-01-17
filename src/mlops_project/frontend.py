import streamlit as st
import requests
from PIL import Image
import io
import os
from google.cloud import run_v2

@st.cache_resource  
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/mlops-project-data-447409/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND_URL", None)

def classify_image(image, backend_url):
    """Send the image to the backend for classification."""
    files = {'file': ('image.jpg', image, 'image/jpeg')}
    response = requests.post(f"{backend_url}/predict/", files=files, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

def main():
    """Main function of the Streamlit frontend."""
    st.title("Image Classification App")
    
    # Get backend URL using Cloud Run client
    backend_url = "http://localhost:8000" #use get_backend_url() when we have the backend running on the cloud
    if backend_url is None:
        st.error("Backend service not found")
        return
    
    # File uploader
    st.write("Upload an image for classification:")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add a classify button
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Get prediction from backend
                result = classify_image(img_byte_arr, backend_url)
                
                if result is not None:
                    st.success(f"Prediction: Class {result['predicted_class']}")
                else:
                    st.error("Failed to get prediction from the backend")

if __name__ == "__main__":
    main()