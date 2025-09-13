import streamlit as st
import os
import datetime
import tempfile
import json
from fpdf import FPDF
from src.main import VideoKYC, KYCProcessingError
import src.database as db
import cv2
import numpy as np


@st.cache_resource
def get_video_kyc_processor():
    """
    Initializes and returns a cached instance of the VideoKYC processor.
    This prevents re-initializing the model on every app rerun.
    """
    return VideoKYC()


def generate_pdf(run_data):
    """
    Generates a PDF report by decoding image data from bytes and using
    temporary files for PDF creation.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)

    pdf.cell(0, 10, "KYC Verification Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(50, 10, "Timestamp:")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, run_data["timestamp"], ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(50, 10, "Video Name:")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, run_data["video_name"], ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Verification Results", ln=True)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w + pdf.l_margin, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(50, 10, "Face Match Status:")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, run_data["match_status"], ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(50, 10, "Similarity Score:")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"{run_data['similarity']:.2%}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Captured Images", ln=True)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w + pdf.l_margin, pdf.get_y())
    pdf.ln(5)

    face_image_data = run_data.get("face_image")
    id_face_image_data = run_data.get("id_face_image")

    margin = 15
    gap = 10
    available_width = pdf.w - (2 * margin) - gap
    max_width_per_image = available_width / 2

    y_before_images = pdf.get_y()
    max_image_height = 0

    x_pos1 = margin
    if face_image_data:
        try:
            nparr = np.frombuffer(face_image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                h, w, _ = img.shape
                aspect_ratio = h / w
                display_height = max_width_per_image * aspect_ratio
                max_image_height = max(max_image_height, display_height)

                pdf.set_font("Arial", "B", 12)
                pdf.text(x_pos1, y_before_images, "User Face")

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as temp_img:
                    temp_img.write(face_image_data)
                    temp_img_path = temp_img.name

                pdf.image(
                    temp_img_path,
                    x=x_pos1,
                    y=y_before_images + 5,
                    w=max_width_per_image,
                )
                os.remove(temp_img_path)
            else:
                pdf.text(x_pos1, y_before_images + 5, "Error decoding user image.")
        except Exception as e:
            pdf.text(x_pos1, y_before_images + 5, f"Error with image: {e}")

    x_pos2 = margin + max_width_per_image + gap
    if id_face_image_data:
        try:
            nparr = np.frombuffer(id_face_image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                h, w, _ = img.shape
                aspect_ratio = h / w
                display_height = max_width_per_image * aspect_ratio
                max_image_height = max(max_image_height, display_height)

                pdf.set_font("Arial", "B", 12)
                pdf.text(x_pos2, y_before_images, "Face from ID")

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as temp_img:
                    temp_img.write(id_face_image_data)
                    temp_img_path = temp_img.name

                pdf.image(
                    temp_img_path,
                    x=x_pos2,
                    y=y_before_images + 5,
                    w=max_width_per_image,
                )
                os.remove(temp_img_path)
            else:
                pdf.text(x_pos2, y_before_images + 5, "Error decoding ID image.")
        except Exception as e:
            pdf.text(x_pos2, y_before_images + 5, f"Error with image: {e}")

    if max_image_height > 0:
        pdf.set_y(y_before_images + 5 + max_image_height + 10)
    else:
        pdf.set_y(y_before_images + 15)
    # --- END: DYNAMIC IMAGE PLACEMENT ---

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "OCR Data from ID", ln=True)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w + pdf.l_margin, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    ocr_data = run_data["ocr_data"]
    if isinstance(ocr_data, str):
        try:
            ocr_data = json.loads(ocr_data)
        except json.JSONDecodeError:
            pdf.cell(0, 10, "Error: Could not decode OCR data.", ln=True)
            ocr_data = {}

    if isinstance(ocr_data, dict) and ocr_data:
        for key, value in ocr_data.items():
            pdf.set_font("Arial", "B", 11)
            pdf.cell(50, 8, f"{str(key).replace('_', ' ').title()}:")
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 8, str(value))
    else:
        pdf.multi_cell(0, 10, "No OCR data extracted.")

    return pdf.output(dest="S").encode("latin-1")


video_kyc_processor = get_video_kyc_processor()
db.init_db()

st.set_page_config(page_title="Video KYC Automation", layout="wide")

st.title("📹 Video KYC Automation")
st.markdown(
    "Upload a short video of a user showing their ID card and their face for automated verification."
)

tab1, tab2 = st.tabs(["KYC Processor", "Run History"])

with tab1:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.video(video_path, width=350)

            if st.button("Process Video", type="primary"):
                with st.spinner("Processing video... This may take a moment."):
                    try:
                        (
                            ocr_result,
                            face_similarity,
                            is_match,
                            best_face_crop,
                            id_face_crop,
                        ) = video_kyc_processor.kyc(video_path)

                        st.success("Processing Complete!")
                        if not ocr_result:
                            st.warning(
                                "⚠️ Verification Incomplete: Could not extract text data (OCR/MRZ) from the ID card. Face verification results are shown below."
                            )

                        face_image_bytes = None
                        id_face_image_bytes = None

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("✅ Verification")
                            match_status = "Matched" if is_match else "Not Matched"
                            st.metric(label="Face Match Status", value=match_status)
                            st.metric(
                                label="Similarity Score", value=f"{face_similarity:.2%}"
                            )

                        with col2:
                            st.subheader("👤 User Face")
                            if best_face_crop is not None:
                                st.image(
                                    cv2.cvtColor(best_face_crop, cv2.COLOR_BGR2RGB)
                                )
                                # Encode numpy array to bytes for DB storage
                                _, buffer = cv2.imencode(".png", best_face_crop)
                                face_image_bytes = buffer.tobytes()
                            else:
                                st.warning("No user face found.")

                        with col3:
                            st.subheader("🆔 Face from ID")
                            if id_face_crop is not None:
                                st.image(cv2.cvtColor(id_face_crop, cv2.COLOR_BGR2RGB))
                                # Encode numpy array to bytes for DB storage
                                _, buffer = cv2.imencode(".png", id_face_crop)
                                id_face_image_bytes = buffer.tobytes()
                            else:
                                st.warning("No face found on ID.")

                        st.divider()
                        st.subheader("📄 OCR Data from ID")
                        st.json(
                            ocr_result if ocr_result else "Could not extract OCR data."
                        )

                        # Prepare data with image bytes for database insertion
                        run_data = {
                            "timestamp": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "video_name": uploaded_file.name,
                            "match_status": match_status,
                            "similarity": face_similarity,
                            "ocr_data": ocr_result or {},
                            "face_image": face_image_bytes,
                            "id_face_image": id_face_image_bytes,
                        }
                        db.add_run(run_data)
                    except KYCProcessingError as e:
                        st.warning(f"⚠️ {e}")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")

with tab2:
    st.subheader("📜 Past Verification Runs")
    history = db.get_all_runs()

    if not history:
        st.info(
            "No runs have been recorded yet. Process a video in the 'KYC Processor' tab."
        )
    else:
        for i, run in enumerate(history):
            with st.expander(f"**{run['timestamp']}** - {run['video_name']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.metric("Face Match Status", run["match_status"])
                    st.metric("Similarity Score", f"{run['similarity']:.2%}")
                    st.write("**OCR Data:**")
                    st.json(run["ocr_data"])

                with col2:
                    st.write("")
                    st.write("")
                    pdf_bytes = generate_pdf(run)
                    safe_timestamp = (
                        run["timestamp"].replace(" ", "_").replace(":", "-")
                    )
                    st.download_button(
                        label="⬇️ Download Report",
                        data=pdf_bytes,
                        file_name=f"KYC_Report_{safe_timestamp}.pdf",
                        mime="application/pdf",
                        key=f"pdf-{i}",
                    )
