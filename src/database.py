import sqlite3
import json
from typing import List, Dict, Any
from src.logger import get_logger

DB_NAME = "kyc_history.db"
logger = get_logger(__name__)


def init_db():
    """Initializes the database.

    The 'runs' table is created if it doesn't exist, with columns for
    timestamp, video name, match status, similarity, OCR data (JSON string),
    and two BLOB columns for face images.

    Returns:
        None
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            video_name TEXT NOT NULL,
            match_status TEXT NOT NULL,
            similarity REAL NOT NULL,
            ocr_data TEXT NOT NULL,
            face_image BLOB,
            id_face_image BLOB
        )
    """
    )

    conn.commit()
    conn.close()
    logger.info("Database initialized...")


def add_run(run_data: Dict[str, Any]):
    """Adds a new run record to the database.

    Args:
        run_data (Dict[str, Any]): A dictionary containing the run information.
            Expected keys:
            - "timestamp" (str): The timestamp of the run.
            - "video_name" (str): The name of the video processed.
            - "match_status" (str): The status of the face match (e.g., "matched", "unmatched").
            - "similarity" (float): The similarity score between the faces.
            - "ocr_data" (Dict[str, Any]): The OCR extracted data from the ID card.
            - "face_image" (bytes, optional): The byte data of the detected face image. Defaults to None.
            - "id_face_image" (bytes, optional): The byte data of the face image from the ID. Defaults to None.

    Returns:
        None
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO runs (timestamp, video_name, match_status, similarity, ocr_data, face_image, id_face_image)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            run_data["timestamp"],
            run_data["video_name"],
            run_data["match_status"],
            run_data["similarity"],
            json.dumps(run_data["ocr_data"]),
            run_data.get("face_image"),
            run_data.get("id_face_image"),
        ),
    )

    conn.commit()
    conn.close()


def get_all_runs() -> List[Dict[str, Any]]:
    """Fetches all run records from the database.

    The 'face_image' and 'id_face_image' fields in the returned dictionaries
    will contain raw byte data from the BLOB columns. The 'ocr_data' field
    will be deserialized from a JSON string back into a dictionary.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a run record.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    runs = []
    for row in rows:
        run_dict = dict(row)
        run_dict["ocr_data"] = json.loads(run_dict["ocr_data"])
        runs.append(run_dict)

    return runs
