import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")

class Settings:
    def __init__(self):
        self.api_base_url = os.getenv("NUCLEO_API_BASE_URL", "https://bnj2vb4zpt.us-east-1.awsapprunner.com").rstrip("/")
        self.api_token = os.getenv("NUCLEO_API_TOKEN", "")
        self.port = int(os.getenv("PORT", "8000"))
        self.host = os.getenv("HOST", "127.0.0.1")
        self.data_dir = DATA_DIR
        self.default_dicom_zip = DATA_DIR / "case-001-dicom.zip"
        self.default_reference_nii = DATA_DIR / "case-001-reference-lesions.nii.gz"
        self.cached_model_nii = DATA_DIR / "lesions.nii.gz"

    @property
    def masked_token(self) -> str:
        if not self.api_token:
            return ""
        if len(self.api_token) <= 12:
            return "***"
        return f"{self.api_token[:8]}...{self.api_token[-6:]}"

    def update(self, base_url: str = None, token: str = None):
        if base_url is not None:
            self.api_base_url = base_url.strip().rstrip("/")
        if token is not None:
            self.api_token = token.strip()

settings = Settings()
