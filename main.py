from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime
from contextlib import contextmanager
import uvicorn
import traceback

# Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "/tmp/inventory.db")
EXCEL_FILE_NAME = "inventory.xlsx"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")


# Models
class Reagent(BaseModel):
    id: int
    name: str
    supplier_code: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    location: Optional[str] = None
    sublocation: Optional[str] = None
    status: Optional[str] = None
    quantity: float = 0.0
    unit: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[str] = None
    last_updated: str

    class Config:
        from_attributes = True


class HistoryEntry(BaseModel):
    id: int
    reagent_id: int
    user: str
    change: float
    notes: Optional[str] = None
    timestamp: str


class QuantityUpdate(BaseModel):
    user: str
    change: float
    notes: Optional[str] = None


# Database utilities
@contextmanager
def get_db_connection():
    """Ensure database connection with proper error handling"""
    conn = None
    try:
        # Create database in current working directory
        db_dir = os.path.dirname(DATABASE_PATH)
        if db_dir:  # Only create directory if path contains subdirectories
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(DATABASE_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


def initialize_database():
    """Initialize database tables"""
    try:
        # This will now create the database in Render's writable space
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reagents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                supplier_code TEXT,
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

            conn.commit()
        logger.info(f"Database initialized successfully at {DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


def excel_to_db():
    """Import data from Excel to database"""
    try:
        if not os.path.exists(EXCEL_FILE_NAME):
            logger.warning(f"Excel file {EXCEL_FILE_NAME} not found!")
            return {"warning": "Excel file not found"}

        df = pd.read_excel(EXCEL_FILE_NAME)
        df = df.where(pd.notnull(df), None)  # Convert NaN to None

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Clear existing data
            cursor.execute("DELETE FROM reagents")

            # Insert new data
            for _, row in df.iterrows():
                cursor.execute('''
                INSERT INTO reagents (
                    name, supplier_code, category, type, 
                    location, sublocation, status, quantity, 
                    unit, notes, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('name'),
                    row.get('supplier_code'),
                    row.get('category'),
                    row.get('type'),
                    row.get('location'),
                    row.get('sublocation'),
                    row.get('status'),
                    float(row.get('quantity', 0)),
                    row.get('unit'),
                    row.get('notes'),
                    row.get('tags')
                ))

            conn.commit()
            return {"status": "success", "imported": len(df)}
    except Exception as e:
        logger.error(f"Excel import failed: {str(e)}")
        return {"error": str(e)}


# FastAPI app
app = FastAPI(
    title="Lab Inventory API",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# API Endpoints
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "running",
        "app": "Lab Inventory API",
        "version": "1.1.0",
        "time": datetime.now().isoformat(),
        "database_path": DATABASE_PATH
    }


@app.get("/reagents", response_model=Dict[str, List[Reagent]], tags=["Reagents"])
async def get_reagents():
    """Get all reagents with proper error handling"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reagents")
            reagents = [dict(row) for row in cursor.fetchall()]
            return {"reagents": reagents}
    except Exception as e:
        logger.error(f"Error fetching reagents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reagents/{reagent_id}", response_model=Reagent, tags=["Reagents"])
async def get_reagent(reagent_id: int):
    """Get single reagent by ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reagents WHERE id=?", (reagent_id,))
            reagent = cursor.fetchone()
            if not reagent:
                raise HTTPException(status_code=404, detail="Reagent not found")
            return dict(reagent)
    except Exception as e:
        logger.error(f"Error fetching reagent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/reagents/{reagent_id}/quantity", tags=["Reagents"])
async def update_quantity(
        reagent_id: int,
        update: QuantityUpdate
):
    """Update reagent quantity with history tracking"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get current quantity
            cursor.execute("SELECT quantity FROM reagents WHERE id=?", (reagent_id,))
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Reagent not found")

            current_qty = result['quantity']
            new_qty = current_qty + update.change

            # Update reagent
            cursor.execute(
                "UPDATE reagents SET quantity=?, last_updated=? WHERE id=?",
                (new_qty, datetime.now().isoformat(), reagent_id)
            )

            # Add history entry
            cursor.execute(
                """INSERT INTO reagent_history 
                (reagent_id, user, change, notes, timestamp)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    reagent_id,
                    update.user,
                    update.change,
                    update.notes or f"Quantity adjusted by {update.change}",
                    datetime.now().isoformat()
                )
            )

            conn.commit()
            return {
                "status": "success",
                "new_quantity": new_qty,
                "reagent_id": reagent_id
            }
    except Exception as e:
        logger.error(f"Quantity update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reagents/history", response_model=Dict[str, List[HistoryEntry]], tags=["History"])
async def get_history():
    """Get all history entries"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reagent_history ORDER BY timestamp DESC")
            history = [dict(row) for row in cursor.fetchall()]
            return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/force-import", tags=["Admin"])
async def force_import():
    """Force re-import from Excel"""
    return excel_to_db()


# Startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        initialize_database()
        result = excel_to_db()
        logger.info(f"Startup completed: {result}")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)