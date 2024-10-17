from typing import Any
from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    asset_path: str
    data: base64

    def read(self) -> base64:
        return self.data
