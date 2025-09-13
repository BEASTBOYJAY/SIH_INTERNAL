from ultralytics import YOLO
from src.config import settings
from deepface import DeepFace
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


class IDDetector:
    """
    A class for detecting IDs in images using a YOLO model.
    """

    def __init__(self):
        logger.info(
            "Initializing IDDetector...",
            model_path=settings.id_detection_model_path,
            device=settings.device,
        )
        self.model = YOLO(settings.id_detection_model_path).to(settings.device)
        self.model.fuse()
        self._warmup_model()

    def _warmup_model(self):
        """
        Warms up the YOLO model by performing a dummy detection.
        This helps to load the model into memory and optimize subsequent inferences.
        """
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        logger.debug(
            "Detecting IDs with confidence threshold.",
            confidence=settings.id_detection_confidence,
        )
        self.detect(dummy_img)
        logger.info("IDDetector model warmup complete.")

    def detect(self, image):
        """
        Detects IDs in the image and returns the raw YOLO results.

        Args:
            image (np.ndarray): The input image in which to detect IDs.

        Returns:
            ultralytics.engine.results.Results: The raw YOLO detection results.
        """

        return self.model(
            image,
            verbose=False,
            conf=settings.id_detection_confidence,
            iou=settings.id_detection_iou,
        )


class FaceDetector:
    """
    A class for detecting faces in images using a YOLO model.
    """

    def __init__(self):
        logger.info(
            "Initializing FaceDetector...",
            model_path=settings.face_detection_model_path,
            device=settings.device,
        )
        self.model = YOLO(settings.face_detection_model_path).to(settings.device)
        self.model.fuse()
        self._warmup_model()

    def _warmup_model(self):
        """
        Warms up the YOLO model by performing a dummy detection.
        This helps to load the model into memory and optimize subsequent inferences.
        """
        logger.info("Warming up FaceDetector model...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        logger.debug(
            "Detecting faces with confidence threshold.",
            confidence=settings.face_detection_confidence,
        )
        self.detect(dummy_img)
        logger.info("FaceDetector model warmup complete.")

    def detect(self, image):
        """
        Detects faces in the image and returns the raw YOLO results.

        Args:
            image (np.ndarray): The input image in which to detect faces.

        Returns:
            ultralytics.engine.results.Results: The raw YOLO detection results.
        """

        return self.model(image, verbose=False, conf=settings.face_detection_confidence)


class FaceRecognition:
    def __init__(self):
        """
        A class for performing face recognition using DeepFace.
        """
        logger.info("Initializing FaceRecognition...")
        self._warmup_deepface_model()

    def _warmup_deepface_model(self):
        """
        Warms up the DeepFace models (Facenet512 and ArcFace) by generating dummy embeddings.
        """
        logger.info("Warming up DeepFace models...")
        zeros_img = np.zeros((112, 112, 3))
        for model_name in ["Facenet512", "ArcFace"]:
            try:
                logger.debug("Warming up model.", model_name=model_name)
                DeepFace.represent(
                    zeros_img, model_name=model_name, detector_backend="skip"
                )
            except Exception as e:
                logger.warning(
                    "Failed to warm up DeepFace model.",
                    model_name=model_name,
                    error=str(e),
                )
        logger.info("DeepFace models warmup complete.")

    def _generate_embedding(self, image):
        """
        Generates an ensemble embedding for a given face image using multiple DeepFace models.
        The embeddings from different models are averaged and normalized.

        Args:
            image (np.ndarray): The face image for which to generate the embedding.

        Returns:
            np.ndarray: The normalized ensemble embedding.
        """
        logger.debug("Generating face embedding...")
        ensemble_embeddings = []
        for model_name in ["Facenet512", "ArcFace"]:
            try:
                embedding_obj = DeepFace.represent(
                    image,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )
                if embedding_obj and len(embedding_obj) > 0:
                    ensemble_embeddings.append(np.array(embedding_obj[0]["embedding"]))
                    logger.debug("Generated embedding.", model_name=model_name)
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    "Could not generate embedding for model.",
                    model_name=model_name,
                    error=str(e),
                )
                continue
        if not ensemble_embeddings:
            logger.warning("No embeddings were generated from any model.")
            return np.zeros(512)
        avg_embedding = np.mean(ensemble_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm < 1e-5:
            return avg_embedding
        logger.debug("Generated and normalized average embedding.")
        return avg_embedding / norm

    def _cosine_similarity(self, embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): The first embedding.
            embedding2 (np.ndarray): The second embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        return np.dot(embedding1, embedding2)

    def recognize(self, image1, image2):
        """
        Performs face recognition by comparing two face images.

        Args:
            image1 (np.ndarray): The first face image.
            image2 (np.ndarray): The second face image.

        Returns:
            Tuple[float, bool]: A tuple containing the similarity score and a boolean
                                indicating if the faces are matched based on the threshold.
        """
        logger.info("Performing face recognition...")
        embedding1 = self._generate_embedding(image1)
        embedding2 = self._generate_embedding(image2)
        similarity = self._cosine_similarity(embedding1, embedding2)
        matched = False
        if similarity > settings.face_recognition_threshold:
            matched = True
        logger.info(
            "Face recognition complete.",
            similarity=float(similarity),
            threshold=settings.face_recognition_threshold,
            matched=matched,
        )

        return similarity, matched
