import streamlit as st
import requests

st.set_page_config(page_title="CyberGuard - Phishing Detector", layout="centered")

st.title("🛡️ CyberGuard - Phishing Detection App")
st.markdown("**Enter a few technical details about the URL and check if it’s legit or a phishing site.**")

# Input form
with st.form("url_form"):
    col1, col2 = st.columns(2)

    with col1:
        url_length = st.slider("URL Length", 0, 300, 50)
        similarity = st.slider("URL Similarity Index", 0.0, 100.0, 25.0)
        external_ref = st.slider("External References", 0, 200, 10)
        lines_of_code = st.slider("HTML Lines of Code", 0, 500, 100)
        https = st.radio("Is HTTPS Enabled?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        largest_line = st.slider("Length of Longest HTML Line", 0, 1000, 300)

    with col2:
        self_ref = st.slider("Self References", 0, 50, 5)
        num_images = st.slider("Number of Images", 0, 50, 5)
        num_js = st.slider("JavaScript Files", 0, 50, 5)
        num_css = st.slider("CSS Files", 0, 50, 3)
        has_social = st.radio("Links to Social Networks?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        has_copyright = st.radio("Copyright Info Present?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        has_description = st.radio("Has Meta Description?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("Predict")

if submitted:
    user_data = {
        'URLLength': url_length,
        'URLSimilarityIndex': similarity,
        'NoOfExternalRef': external_ref,
        'LineOfCode': lines_of_code,
        'IsHTTPS': https,
        'NoOfSelfRef': self_ref,
        'NoOfImage': num_images,
        'NoOfJS': num_js,
        'NoOfCSS': num_css,
        'HasSocialNet': has_social,
        'HasCopyrightInfo': has_copyright,
        'LargestLineLength': largest_line,
        'HasDescription': has_description
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=user_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
            st.info(f"Confidence: **{result['confidence']*100:.2f}%**")
            if result['prediction'] == "Phishing":
                st.warning("⚠️ This URL looks suspicious.")
            else:
                st.success("✅ This URL appears to be safe.")
        else:
            st.error("Something went wrong while getting a prediction.")

    except Exception as e:
        st.error(f"❌ Could not connect to backend: {e}")

st.divider()
st.caption("🔬 Built with Flask + Streamlit + scikit-learn | Poorva Pathak")
