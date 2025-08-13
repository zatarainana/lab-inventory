from fastapi import FastAPI, HTTPException, Body, Depends
import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime
import traceback
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import uvicorn
from contextlib import contextmanager
from pydantic import BaseModel
from pathlib import Path

# --- Configuration ---
# Use environment variable for database path or fallback to local file
DATABASE_PATH = os.getenv("DATABASE_PATH", "inventory.db")
EXCEL_FILE_NAME = "inventory.xlsx"


# --- Models ---
class ReagentBase(BaseModel):
    name: str
    supplier_code: Optional[str] = None
    datasheet_url: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    location: Optional[str] = None
    sublocation: Optional[str] = None
    status: Optional[str] = None
    quantity: float = 0.0
    unit: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[str] = None


class ReagentCreate(ReagentBase):
    pass


class Reagent(ReagentBase):
    id: int
    last_updated: str

    class Config:
        from_attributes = True  # Updated from orm_mode


class HistoryEntry(BaseModel):
    reagent_id: int
    user: str
    change: float
    notes: Optional[str] = None
    timestamp: str = datetime.now().isoformat()


class HistoryRecord(HistoryEntry):
    id: int

    class Config:
        from_attributes = True  # Updated from orm_mode


class QuantityUpdate(BaseModel):
    user: str
    change: float
    notes: Optional[str] = None


# --- Database Utilities ---
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    finally:
        if conn:
            conn.close()


@contextmanager
def get_db_cursor():
    """Context manager for database cursors"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e


def initialize_database():
    """Initialize database with tables"""
    try:
        logger.info("Initializing database...")

        # Ensure directory exists for the database file
        db_dir = os.path.dirname(DATABASE_PATH)
        if db_dir:  # Only try to create if path contains directories
            os.makedirs(db_dir, exist_ok=True)

        with get_db_cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reagents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                supplier_code TEXT,
                datasheet_url TEXT,
                category TEXT,
                type TEXT,
                location TEXT,
                sublocation TEXT,
                status TEXT,
                quantity REAL DEFAULT 0,
                unit TEXT,
                notes TEXT,
                tags TEXT,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reagent_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reagent_id INTEGER NOT NULL,
                user TEXT NOT NULL,
                change REAL NOT NULL,
                notes TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(reagent_id) REFERENCES reagents(id) ON DELETE CASCADE
            )''')

        logger.info(f"Database initialized successfully at {DATABASE_PATH}")
    except Exception as e:
        log_error(f"Database initialization failed: {str(e)}")
        raise


# --- Application Setup ---
app = FastAPI(
    title="Lab Inventory API",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


def log_error(message):
    """Log errors with traceback"""
    logger.error(message)
    logger.error(traceback.format_exc())


# --- Data Processing ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the input data"""
    df = df.fillna('')
    required_columns = ['name']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)

    return df.replace('', None)


def excel_to_db() -> Dict[str, Any]:
    """Import data from Excel to database"""
    try:
        if not os.path.exists(EXCEL_FILE_NAME):
            logger.warning(f"Excel file {EXCEL_FILE_NAME} not found!")
            return {"warning": "Excel file not found"}

        df = pd.read_excel(EXCEL_FILE_NAME)
        df = clean_data(df)
        df['last_updated'] = datetime.now().isoformat()

        success_count = 0
        error_count = 0
        error_messages = []

        with get_db_cursor() as cursor:
            for _, row in df.iterrows():
                try:
                    row_dict = {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
                    columns = ', '.join(row_dict.keys())
                    placeholders = ', '.join(['?'] * len(row_dict))
                    sql = f"INSERT OR REPLACE INTO reagents ({columns}) VALUES ({placeholders})"
                    cursor.execute(sql, tuple(row_dict.values()))
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    error_messages.append(f"Error inserting row {_}: {str(e)}")

        return {
            "total_records": len(df),
            "successful": success_count,
            "failed": error_count,
            "errors": error_messages if error_count > 0 else None
        }
    except Exception as e:
        log_error(f"Excel import failed: {str(e)}")
        return {"error": str(e)}


# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    try:
        initialize_database()
        result = excel_to_db()
        if isinstance(result, dict) and result.get('error'):
            if result.get('successful', 0) > 0:
                logger.warning("Started with partial data")
            else:
                raise RuntimeError("Failed to import any data")
    except Exception as e:
        log_error(f"Startup failed: {str(e)}")
        raise


# --- API Endpoints ---
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "running",
        "app": "Lab Inventory API",
        "version": "1.1.0",
        "time": datetime.now().isoformat(),
        "database_path": DATABASE_PATH
    }


# [Rest of your endpoint implementations remain the same...]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)