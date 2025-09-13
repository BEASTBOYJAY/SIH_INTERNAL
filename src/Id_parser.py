import numpy as np
from typing import List, Dict, Any
import datetime
import base64
import cv2
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.models import IDCardInfo
from src.prompt_template import PROMPT
from src.logger import get_logger
from src.config import settings

logger = get_logger()


class IDParser:
    """
    A class responsible for parsing information from Aadhar ID cards using a Large Language Model (LLM).
    It leverages Google's Gemini model and Langchain for image processing and data extraction.
    """

    def __init__(self):
        try:

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", google_api_key=settings.gemini_api_key
            )
            logger.info("Successfully initialized LLM...")
        except Exception as e:
            logger.error("Failed to initialize LLM", error=str(e), exc_info=True)
            raise ValueError("Failed to initialize LLM")
        self.parser = JsonOutputParser(pydantic_object=IDCardInfo)
        self.chain = PROMPT | llm | self.parser

    def _parse_id_using_llm(
        self, front_id: np.ndarray, back_id: np.ndarray
    ) -> Dict[str, Any]:
        """
        Parses information from the front and back images of an ID card using an LLM.
        The images are first converted to base64 strings before being sent to the LLM.

        Args:
            front_id (np.ndarray): The numpy array representing the front image of the ID card.
            back_id (np.ndarray): The numpy array representing the back image of the ID card.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed information from the ID card.
                            Returns an empty dictionary if parsing fails.
        """
        _, buffer = cv2.imencode(".jpg", front_id)
        front_id_base64 = base64.b64encode(buffer).decode("utf-8")

        _, buffer = cv2.imencode(".jpg", back_id)
        back_id_base64 = base64.b64encode(buffer).decode("utf-8")

        try:
            response = self.chain.invoke(
                {
                    "front_image_data": front_id_base64,
                    "back_image_data": back_id_base64,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
            return response
        except Exception as e:
            logger.error("Failed to parse id", error=str(e), exc_info=True)
            return {}

    def process_adhar_card(
        self, front_id_image: np.ndarray, back_id_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Processes the Aadhar card images to extract information.

        Args:
            front_id_image (np.ndarray): The numpy array representing the front image of the Aadhar card.
            back_id_image (np.ndarray): The numpy array representing the back image of the Aadhar card.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted Aadhar card information.
        """
        llm_response = self._parse_id_using_llm(front_id_image, back_id_image)
        return llm_response
