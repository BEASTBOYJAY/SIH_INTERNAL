import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly specialized AI assistant for extracting information from Indian Aadhar Cards. "
            "You will be given two images: the front and the back of the card. "
            "Your task is to analyze both images and accurately extract the key details into the specified format. "
            "Ensure that all extracted text is in English. Pay close attention to spelling, dates, and numbers to ensure perfect accuracy.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            [
                {
                    "type": "text",
                    "text": "Here are the front and back images of an Aadhar card. Please extract the required information.",
                },
                {
                    "type": "text",
                    "text": "--- FRONT IMAGE ---",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{front_image_data}",
                },
                {
                    "type": "text",
                    "text": "--- BACK IMAGE ---",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{back_image_data}",
                },
            ],
        ),
    ]
)
