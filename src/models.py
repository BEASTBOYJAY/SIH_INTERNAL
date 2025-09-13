from pydantic import BaseModel, Field
from typing import List


class IDCardInfo(BaseModel):
    """Extracted information from an Aadhar card."""

    name: str = Field(description="The full name of the cardholder.")
    dob: str = Field(description="The date of birth in DD/MM/YYYY format.")
    gender: str = Field(
        description="The gender of the cardholder (e.g., Male, Female)."
    )
    aadhar_number: str = Field(
        description="The 12-digit unique Aadhar number, often formatted as XXXX XXXX XXXX."
    )
    address: str = Field(
        description="The full residential address from the back of the card."
    )
