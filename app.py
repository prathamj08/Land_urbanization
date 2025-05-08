import streamlit as st
import base64
import numpy as np
import json
import os
from analysis import analyze_image, plot_change_map, save_fig_to_bytes
from streamlit_lottie import st_lottie

# --- Page Config ---
st.set_page_config(page_title="🌍 Land Cover Change Detection", layout="wide", initial_sidebar_state="expanded")

# --- Load Local Lottie Animation ---
def load_lottie_file(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

# Load from local JSON animation file
satellite_lottie = load_lottie_file("Animation - 1746611268183.json")

# --- Custom Styling ---
st.markdown("""
    <style>
        body {
            background-color: #1E1E2F;
            color: #FFFFFF;
        }
        .block-container {
            padding: 2rem 4rem;
            background-color: #1E1E2F;
        }
        h1, h2, h3, h4 {
            color: #F1C40F;
        }
        .stButton>button {
            background-color: #27AE60;
            color: white;
            font-weight: 600;
            border-radius: 8px;
        }
        .stDownloadButton>button {
            background-color: #2980B9;
            color: white;
            border-radius: 6px;
        }
        .metric-container {
            padding: 1rem;
            border: 1px solid #444;
            border-radius: 8px;
            background-color: #2C2C3B;
            margin-bottom: 1rem;
            transition: 0.3s ease-in-out;
        }
        .metric-container:hover {
            transform: scale(1.02);
            box-shadow: 0 0 10px rgba(255,255,255,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Logo and Title ---
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

logo_base64 = get_image_base64("/home/pratham/Music/download2.png")
st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 20px; padding-bottom: 10px;'>
        <img src='data:image/png;base64,{logo_base64}' width='60' />
        <h1 style='margin: 0;'>🌍 Land Cover Change Detection</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("Detect vegetation, urban expansion, and environmental changes between two satellite images using advanced indices (NDVI, NDBI, ExG, VARI).")
with st.expander("ℹ️  What do these indices mean? Click to learn more"):
    st.markdown("""
    <style>
        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #2C2C3B;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .info-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #F1C40F;
        }
        .info-text {
            font-size: 16px;
            line-height: 1.5;
            color: #DDDDDD;
        }
        .info-highlight {
            font-weight: bold;
            color: #58D68D;
        }
    </style>

    <div class="info-box">
        <div class="info-title">🟢 NDVI — Normalized Difference Vegetation Index</div>
        <div class="info-text">
            Uses near-infrared and red light to measure vegetation health.<br>
            <span class="info-highlight">High NDVI</span> indicates healthy, green vegetation.<br>
            <span style='color:#EC7063; font-weight:bold;'>Low or negative NDVI</span> indicates urban areas, water, or barren land.
        </div>
    </div>

    <div class="info-box">
        <div class="info-title">🏙️ NDBI — Normalized Difference Built-up Index</div>
        <div class="info-text">
            Highlights built-up regions such as cities, roads, and infrastructure.<br>
            <span class="info-highlight">High NDBI</span> reflects dense human-made surfaces like buildings and concrete.
        </div>
    </div>

    <div class="info-box">
        <div class="info-title">🌿 ExG — Excess Green Index</div>
        <div class="info-text">
            Extracted from visible RGB data, it emphasizes the intensity of green vegetation.<br>
            Useful when near-infrared bands (like NDVI) aren't available.
        </div>
    </div>

    <div class="info-box">
        <div class="info-title">🌾 VARI — Visible Atmospherically Resistant Index</div>
        <div class="info-text">
            Robust against lighting and atmospheric variations.<br>
            Ideal for estimating green cover in variable environments using only visible light.
        </div>
    </div>

    <hr style="border: 0.5px solid #555;">
    <p style="text-align: center; color: #AAAAAA; font-size: 14px;">
        These indices allow detailed detection of environmental changes, urban growth, and vegetation shifts over time.
    </p>
    """, unsafe_allow_html=True)


# --- Animation Display ---
with st.container():
    if satellite_lottie:
        st_lottie(satellite_lottie, height=180, key="satellite")
    else:
        st.warning("⚠️ Animation not loaded. Please check the local file path.")

# --- Upload Sections ---
col1, col2 = st.columns(2)

@st.cache_data
def load_and_analyze_image(uploaded_file):
    if uploaded_file:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            return analyze_image(uploaded_file)
    return None

with col1:
    st.subheader("📥 Upload First Image")
    uploaded_file_1 = st.file_uploader("Upload .img file", type=["img"], key="img1")
    image_data_1 = load_and_analyze_image(uploaded_file_1)

with col2:
    st.subheader("📥 Upload Second Image")
    uploaded_file_2 = st.file_uploader("Upload .img file", type=["img"], key="img2")
    image_data_2 = load_and_analyze_image(uploaded_file_2)

def display_image_analysis(image_data, label="Image"):
    if image_data:
        with st.expander(f"📊 Detailed Analysis for {image_data['filename']}", expanded=True):
            tabs = st.tabs(["🍃 NDVI", "🏙️ NDBI", "🌿 ExG", "🌾 VARI"])
            for i, key in enumerate(['ndvi', 'ndbi', 'exg', 'vari']):
                with tabs[i]:
                    data = image_data.get(key)
                    if data is not None:
                        mean_val = np.nanmean(data)
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{key.upper()} Index</h4>
                            <p style="font-size: 22px; color: #58D68D;"><b>{mean_val:.4f}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        fig = image_data[f"{key}_fig"]
                        st.pyplot(fig)
                        st.download_button(f"📥 Download {key.upper()} Plot", data=save_fig_to_bytes(fig), file_name=f"{key}_{label.lower()}.png")
                        
                    else:
                        st.warning(f"{key.upper()} index not found.")

# --- Show Image Data ---
with col1:
    if image_data_1:
        display_image_analysis(image_data_1, "1st")

with col2:
    if image_data_2:
        display_image_analysis(image_data_2, "2nd")

# --- Change Detection ---
if image_data_1 and image_data_2:
    st.markdown("## 🔍 Change Detection")
    if st.button("🧪 Analyze Differences"):
        with st.spinner("Calculating change metrics..."):
            change_tabs = st.tabs(["🍃 NDVI", "🏙️ NDBI", "🌿 ExG", "🌾 VARI"])
            for i, key in enumerate(['ndvi', 'ndbi', 'exg', 'vari']):
                with change_tabs[i]:
                    before = image_data_1.get(key)
                    after = image_data_2.get(key)
                    if before is not None and after is not None:
                        diff = np.nanmean(after - before)
                        st.metric(label=f"{key.upper()} Mean Change", value=f"{diff:.4f}")

                        if key == "ndvi":
                            threshold = 0.2
                            before_veg = before > threshold
                            after_veg = after > threshold
                            total_pixels = np.count_nonzero(~np.isnan(before) & ~np.isnan(after))
                            gain = np.count_nonzero(after_veg & ~before_veg)
                            loss = np.count_nonzero(before_veg & ~after_veg)
                            gain_pct = (gain / total_pixels * 100) if total_pixels else 0
                            loss_pct = (loss / total_pixels * 100) if total_pixels else 0

                            col_gain, col_loss = st.columns(2)
                            col_gain.metric("🌱 Vegetation Gain", f"{gain_pct:.2f}%")
                            col_loss.metric("🔥 Vegetation Loss", f"{loss_pct:.2f}%")

                        # fig = plot_change_map(before, after, title=f"{key.upper()} Change")
                        # st.pyplot(fig)
                        fig = plot_change_map(before, after, title="")  # remove in-figure title
                        fig.set_size_inches(6, 4)  # resize figure to make it smaller

                        st.pyplot(fig)
                        st.markdown("<p style='text-align:center; font-size: 14px; color: #ccc;'>🔵 Blue = Loss &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Red = Gain</p>", unsafe_allow_html=True)

                        st.download_button(
                            f"📥 Download {key.upper()} Change Map",
                            data=save_fig_to_bytes(fig),
                            file_name=f"{key}_change.png"
                        )
                    else:
                        st.info(f"No data for {key.upper()}")



# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: gray;'>Built with ❤️ by Team EarthVision • <a style='color:#3498DB' href='https://github.com/your-repo' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)
