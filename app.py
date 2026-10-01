import streamlit as st
import cv2
import numpy as np
import joblib

from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tomato Guard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #e8f5e9,
        #f1f8e9
    );
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
    color: #52635a;
}

.card {
    padding: 1.3rem;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.result-card {
    padding: 2rem;
    border-radius: 20px;
    background: #f1f8e9;
    border: 1px solid #c8e6c9;
}

.big-result {
    font-size: 30px;
    font-weight: 800;
}

.confidence {
    font-size: 42px;
    font-weight: 800;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
}

.small-label {
    color: #68756d;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FEATURE EXTRACTION
# SAME PIPELINE USED DURING TRAINING
# =========================================================

def extract_features(image):

    # Resize
    image = cv2.resize(image, (224, 224))

    # -----------------------------------------------------
    # HSV - 6 FEATURES
    # -----------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hsv_features = [
        np.mean(h),
        np.std(h),
        np.mean(s),
        np.std(s),
        np.mean(v),
        np.std(v)
    ]

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # LBP - 10 FEATURES
    # -----------------------------------------------------

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    n_bins = points + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    lbp_hist = lbp_hist.astype(float)

    lbp_hist /= (
        lbp_hist.sum() + 1e-7
    )

    # -----------------------------------------------------
    # GLCM - 4 FEATURES
    # -----------------------------------------------------

    gray_small = (
        gray / 32
    ).astype(np.uint8)

    glcm = graycomatrix(
        gray_small,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    )[0, 0]

    correlation = graycoprops(
        glcm,
        "correlation"
    )[0, 0]

    energy = graycoprops(
        glcm,
        "energy"
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    )[0, 0]

    glcm_features = [
        contrast,
        correlation,
        energy,
        homogeneity
    ]

    # -----------------------------------------------------
    # TOTAL = 20 FEATURES
    # -----------------------------------------------------

    features = np.concatenate([
        hsv_features,
        lbp_hist,
        glcm_features
    ])

    return features


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/random_forest_model.pkl"
    )


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌿 Tomato Guard AI")

    st.write(
        "AI-powered tomato leaf health monitoring"
    )

    st.divider()

    st.markdown("### ⚙️ System Information")

    st.write("🤖 Model: Random Forest")
    st.write("🔬 Features: 20")
    st.write("🎨 HSV: 6")
    st.write("🧩 LBP: 10")
    st.write("📊 GLCM: 4")

    st.divider()

    st.markdown("### 📈 Model Performance")

    st.metric(
        "Test Accuracy",
        "85.87%"
    )

    st.caption(
        "Performance measured on the held-out test dataset."
    )

    st.divider()

    st.success("🟢 AI Model Ready")


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🌿 Tomato Guard AI
</div>

<div class="hero-subtitle">
Smart Tomato Crop Health Monitoring System
</div>

<p>
Upload a tomato leaf image and let the machine-learning
model analyze its visual characteristics.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze Leaf",
    "🧠 How It Works",
    "📊 Model Performance",
    "ℹ️ About Project"
])


# =========================================================
# TAB 1 - ANALYSIS
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        '📸 Analyze Your Tomato Leaf'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a tomato leaf."
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload a tomato leaf image to begin analysis."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card">
            <h3>📷 Upload</h3>
            <p>Select a tomato leaf image.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
            <h3>🔬 Analyze</h3>
            <p>Extract visual features using image processing.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
            <h3>🧠 Predict</h3>
            <p>Random Forest predicts the class.</p>
            </div>
            """, unsafe_allow_html=True)

    else:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with col1:

            st.markdown(
                "### 📷 Your Leaf"
            )

            st.image(
                image_rgb,
                use_container_width=True
            )

        with col2:

            st.markdown(
                "### 🔬 Image Analysis"
            )

            info1, info2 = st.columns(2)

            with info1:
                st.metric(
                    "Input Type",
                    uploaded_file.type
                )

            with info2:
                st.metric(
                    "Processed Size",
                    "224 × 224"
                )

            st.write("")

            st.markdown("""
            **Processing pipeline**

            🎨 HSV color analysis  
            🧩 LBP texture analysis  
            📊 GLCM texture analysis  

            **20 numerical features**
            """)

            if st.button(
                "🔍 Analyze Leaf",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "🔬 Analyzing leaf characteristics..."
                ):

                    features = extract_features(
                        image
                    )

                    features = features.reshape(
                        1, -1
                    )

                    probabilities = (
                        model.predict_proba(
                            features
                        )[0]
                    )

                    class_names = model.classes_

                    top_indices = np.argsort(
                        probabilities
                    )[::-1][:3]

                    predicted_class = (
                        class_names[
                            top_indices[0]
                        ]
                    )

                    confidence = (
                        probabilities[
                            top_indices[0]
                        ] * 100
                    )

                st.success(
                    "✅ Analysis completed!"
                )

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 🧠 AI Prediction"
                )

                st.markdown(
                    f'<div class="big-result">'
                    f'{predicted_class}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'{confidence:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Predicted probability"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write("")

                st.markdown(
                    "### 🏆 Top 3 Predictions"
                )

                for rank, index in enumerate(
                    top_indices,
                    start=1
                ):

                    probability = (
                        probabilities[index]
                        * 100
                    )

                    st.write(
                        f"**{rank}. "
                        f"{class_names[index]}**"
                    )

                    st.progress(
                        float(
                            probabilities[index]
                        )
                    )

                    st.caption(
                        f"{probability:.2f}%"
                    )

                with st.expander(
                    "🔬 View 20 extracted features"
                ):

                    st.write(
                        "The model receives 20 numerical "
                        "features extracted from the image."
                    )

                    st.dataframe(
                        features,
                        use_container_width=True
                    )

                st.warning(
                    "⚠️ This is an AI-based prediction "
                    "and should not be treated as a "
                    "definitive plant diagnosis."
                )


# =========================================================
# TAB 2 - HOW IT WORKS
# =========================================================

with tab2:

    st.markdown(
        "## 🧠 How Does Tomato Guard AI Work?"
    )

    st.write(
        "The system converts the uploaded leaf image "
        "into numerical features before classification."
    )

    steps = [
        (
            "1️⃣",
            "Image Input",
            "The user uploads a tomato leaf image."
        ),
        (
            "2️⃣",
            "Image Processing",
            "The image is resized to 224 × 224 pixels."
        ),
        (
            "3️⃣",
            "Color Analysis",
            "HSV statistics provide 6 color features."
        ),
        (
            "4️⃣",
            "Texture Analysis",
            "LBP provides 10 texture features."
        ),
        (
            "5️⃣",
            "Texture Analysis",
            "GLCM provides 4 additional texture features."
        ),
        (
            "6️⃣",
            "Feature Vector",
            "All features are combined into 20 values."
        ),
        (
            "7️⃣",
            "Classification",
            "Random Forest predicts the leaf class."
        )
    ]

    for icon, title, description in steps:

        st.markdown(
            f"""
            <div class="card">
            <h3>{icon} {title}</h3>
            <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")


# =========================================================
# TAB 3 - MODEL PERFORMANCE
# =========================================================

with tab3:

    st.markdown(
        "## 📊 Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Random Forest",
            "85.87%"
        )

        st.caption(
            "Test Accuracy"
        )

    with col2:

        st.metric(
            "SVM",
            "73.99%"
        )

        st.caption(
            "Test Accuracy"
        )

    with col3:

        st.metric(
            "Best Model",
            "Random Forest"
        )

        st.caption(
            "Selected for prediction"
        )

    st.divider()

    st.markdown(
        "### 🏆 Model Comparison"
    )

    st.bar_chart(
        {
            "Random Forest": 85.87,
            "SVM": 73.99
        }
    )

    st.info(
        "Random Forest was selected because it achieved "
        "higher test accuracy than the SVM model."
    )


# =========================================================
# TAB 4 - ABOUT
# =========================================================

with tab4:

    st.markdown(
        "## 🌱 About the Project"
    )

    st.write(
        "Tomato Guard AI is an AI-based crop health "
        "monitoring system designed to classify tomato "
        "leaf conditions using image processing and "
        "machine learning."
    )

    st.markdown("### 🛠️ Technologies")

    technologies = [
        "🐍 Python",
        "📷 OpenCV",
        "🔬 Scikit-image",
        "🌲 Random Forest",
        "📊 NumPy",
        "🎨 Streamlit"
    ]

    for technology in technologies:
        st.write(technology)

    st.markdown("### 🎯 Project Pipeline")

    st.code("""
Dataset
   ↓
Preprocessing
   ↓
HSV + LBP + GLCM
   ↓
20 Features
   ↓
Random Forest
   ↓
Disease Prediction
   ↓
Crop Health Result
    """)

    st.success(
        "🌿 Tomato Guard AI is ready to analyze tomato leaves!"
    )