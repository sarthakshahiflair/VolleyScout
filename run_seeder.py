import logging

from app.db.init_db import init_db
from app.db.seed_db import seed_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Initializing database tables...")
    init_db()
    seed_db()
    logger.info("Database setup complete.")
