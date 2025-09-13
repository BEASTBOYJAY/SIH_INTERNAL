import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

from src.detector_nd_recognizer import IDDetector, FaceDetector, FaceRecognition
from src.Id_parser import IDParser
from src.utils import (
    get_best_quality_crop,
    get_best_quality_face_crop,
    add_padding,
    resize_proportional,
)
from src.logger import get_logger

logger = get_logger(__name__)


class KYCProcessingError(Exception):
    """Custom exception for KYC processing failures."""

    pass


class VideoKYC:
    """
    A class to perform Know Your Customer (KYC) processing from a video input.
    """

    def __init__(self):
        logger.info("Initializing VideoKYC service...")
        self.id_detector = IDDetector()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognition()
        self.id_parser = IDParser()
        self.detected_ids_front = []
        self.detected_ids_back = []
        self.detected_faces = []
        logger.info("VideoKYC service initialized successfully.")

    def _process_video(self, video_path: str):
        """
        Processes the input video to detect IDs (front and back) and faces.
        Detected IDs and faces are stored internally for further processing.

        Args:
        video_path (str): The path to the video file.

        Raises:
        ValueError: If the video file cannot be opened.

        Returns:
        None
        """
        logger.info("Starting video processing.", video_path=video_path)
        self.detected_ids_front.clear()
        self.detected_ids_back.clear()
        self.detected_faces.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video.", video_path=video_path)
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        pbar = tqdm(total=total_frames, desc="Processing Video Frames", unit="frame")
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                pbar.update(1)

                if frame_idx % 4 == 0:
                    frame_for_face_detection = frame.copy()
                    parent_shape = frame.shape
                    try:
                        id_results = self.id_detector.detect(frame)
                    except Exception as e:
                        print(f"ID detection failed at frame {frame_idx}: {e}")
                        continue

                    for result in id_results or []:
                        for box in getattr(result, "boxes", []):
                            coords = box.xyxy.cpu().numpy().astype(int)
                            x1, y1, x2, y2 = coords[0]
                            id_crop = frame[y1:y2, x1:x2]

                            face_results = self.face_detector.detect(id_crop)

                            if len(face_results[0].boxes) > 0:
                                self.detected_ids_front.append(id_crop)
                            else:
                                self.detected_ids_back.append(id_crop)
                            cv2.rectangle(
                                frame_for_face_detection,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 0),
                                -1,
                            )
                    try:
                        face_results = self.face_detector.detect(
                            frame_for_face_detection
                        )
                    except Exception as e:
                        face_results = []
                        print(f"Face detection failed at frame {frame_idx}: {e}")

                    frame_height, frame_width = parent_shape[:2]
                    frame_area = frame_height * frame_width

                    min_face_area = 0.01 * frame_area

                    for result in face_results or []:
                        for box in getattr(result, "boxes", []):
                            coords = box.xyxy.cpu().numpy().astype(int)
                            x1, y1, x2, y2 = coords[0]

                            face_width = x2 - x1
                            face_height = y2 - y1
                            face_area = face_width * face_height

                            if face_area > min_face_area:
                                px1, py1, px2, py2 = add_padding(
                                    x1, y1, x2, y2, parent_shape
                                )
                                face_crop = frame[py1:py2, px1:px2]
                                self.detected_faces.append(face_crop)

                frame_idx += 1
        finally:
            cap.release()
            pbar.close()
            logger.info(
                "Finished processing video.",
                total_frames_processed=frame_idx,
                front_ids_found=len(self.detected_ids_front),
                back_ids_found=len(self.detected_ids_back),
                faces_found=len(self.detected_faces),
            )

    def kyc(self, video_path: str) -> Tuple[
        Optional[Dict[str, Any]],
        float,
        bool,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Performs the full KYC process on a given video.
        This involves detecting IDs and faces, extracting the face from the ID,
        performing face recognition, and parsing ID information.

        Args:
        video_path (str): The path to the video file for KYC.

        Raises:
        KYCProcessingError: If essential components (front ID, back ID, or user face)
        cannot be clearly detected in the video.

        Returns:
        Tuple[Optional[Dict[str, Any]], float, bool, Optional[np.ndarray], Optional[np.ndarray]]:
        A tuple containing:
        - parsed_id_json_content (Optional[Dict[str, Any]]): The extracted information from the ID card, or None if parsing failed.
        - similarity (float): The similarity score between the detected user face and the face from the ID.
        - matched (bool): True if the faces are considered a match based on the similarity threshold, False otherwise.
        - best_face_crop (Optional[np.ndarray]): The best quality detected user face crop, or None.
        - face_crop_from_id (Optional[np.ndarray]): The face crop extracted from the ID card, or None.
        """
        logger.info("Starting KYC process.", video_path=video_path)
        self._process_video(video_path)

        best_front_id_crop = get_best_quality_crop(self.detected_ids_front)
        best_back_id_crop = get_best_quality_crop(self.detected_ids_back)
        best_face_crop = get_best_quality_face_crop(self.detected_faces)

        error_messages = []
        if best_front_id_crop is None:
            error_messages.append("the front of the ID card")
        if best_back_id_crop is None:
            error_messages.append("the back of the ID card")
        if best_face_crop is None:
            error_messages.append("the user's face")

        if error_messages:
            message = f"Could not clearly detect {', '.join(error_messages)}. Please try again, ensuring the items are well-lit and clearly visible in the video."
            logger.warning(message)
            raise KYCProcessingError(message)

        face_crop_from_id = None
        try:
            faceid_results = self.face_detector.detect(best_front_id_crop)
            if not faceid_results or not hasattr(faceid_results[0], "boxes"):
                raise ValueError("No face detected on ID front crop.")

            box_collection = faceid_results[0].boxes

            if len(box_collection) == 0:
                raise ValueError("No bounding boxes found in the detected face result.")
            coords = box_collection.xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = coords
            px1, py1, px2, py2 = add_padding(x1, y1, x2, y2, best_front_id_crop.shape)
            face_crop_from_id = best_front_id_crop[py1:py2, px1:px2]

        except Exception as e:
            logger.error(
                "Failed to extract face from ID card.", error=str(e), exc_info=True
            )
            return None, 0.0, False, None, None

        try:
            similarity, matched = self.face_recognizer.recognize(
                face_crop_from_id, best_face_crop
            )
        except Exception as e:
            logger.error("Face recognition step failed.", error=str(e), exc_info=True)
            similarity, matched = 0.0, False

        parsed_id_json_content = None
        try:
            parsed_id_json_content = self.id_parser.process_adhar_card(
                best_front_id_crop, best_back_id_crop
            )
        except Exception as e:
            logger.error(
                "An unexpected error occurred during ID parsing.",
                error=str(e),
                exc_info=True,
            )
        logger.info(
            "KYC process finished.",
            final_match_status=matched,
            face_similarity=similarity,
        )
        best_face_crop = resize_proportional(best_face_crop, 120)
        face_crop_from_id = resize_proportional(face_crop_from_id, 120)
        return (
            parsed_id_json_content,
            similarity,
            matched,
            best_face_crop,
            face_crop_from_id,
        )
