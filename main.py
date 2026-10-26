from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from app.main import main

if __name__ == "__main__":
    main()
