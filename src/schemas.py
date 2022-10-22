from pydantic import BaseModel, Field


class Txt2ImgInput(BaseModel):
    prompt: str
    guidance: float = Field(7.5, ge=-20, le=20)
    width: int = Field(512, ge=320, le=1024)
    height: int = Field(512, ge=320, le=1024)
    steps: int = Field(50, ge=10, le=150)
    seed: int = None


class Img2ImgInput(BaseModel):
    prompt: str
    guidance: float = Field(7.5, ge=-20, le=20)
    strength: float = Field(0.8, ge=0, le=1.0)
    steps: int = Field(50, ge=10, le=150)
    seed: int = None
