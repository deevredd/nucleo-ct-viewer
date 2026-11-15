#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

def main():
    print("=" * 65)
    print("  Nucleo OmniLesion CT Viewer — AI Engineer Take-Home")
    print("=" * 65)
    print(f"  • API Base URL:  {settings.api_base_url}")
    print(f"  • Token Status:  {'Configured (' + settings.masked_token + ')' if settings.api_token else 'NOT SET'}")
    print(f"  • Local Server:  http://{settings.host}:{settings.port}")
    print("=" * 65)
    print("  Starting server... Press Ctrl+C to stop.\n")

    try:
        webbrowser.open(f"http://{settings.host}:{settings.port}")
    except Exception:
        pass

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False
    )

if __name__ == "__main__":
    main()
