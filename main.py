from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime
from contextlib import contextmanager
import uvicorn

# Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "/tmp/inventory.db")
EXCEL_FILE_NAME = "inventory.xlsx"


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


class HistoryEntry(BaseModel):
    id: int
    reagent_id: int
    user: str
    change: float
    notes: Optional[str] = None
    timestamp: str


# Database setup
@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
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
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS reagent_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reagent_id INTEGER NOT NULL,
            user TEXT NOT NULL,
            change REAL NOT NULL,
            notes TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(reagent_id) REFERENCES reagents(id) ON DELETE CASCADE
        )""")


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
    allow_methods=["*"],
    allow_headers=["*"]
)


# API Endpoints
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "running",
        "app": "Lab Inventory API",
        "version": "1.1.0",
        "time": datetime.now().isoformat()
    }


def get_db_connection():
    pass


@app.get("/reagents", response_model=List[Reagent], tags=["Reagents"])
async def get_reagents():
    """Get all reagents"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reagents")
            reagents = [dict(row) for row in cursor.fetchall()]
            return {"reagents": reagents}  # Wrap in object with reagents key
    except Exception as e:
        log_error(f"Error fetching reagents: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/reagents/history", response_model=Dict[str, List[HistoryEntry]], tags=["History"])
async def get_history():
    try:
        with get_db() as conn:
            history = conn.execute("SELECT * FROM reagent_history").fetchall()
            return {"history": [dict(row) for row in history]}
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        raise HTTPException(500, "Database error")


@app.put("/reagents/{reagent_id}", tags=["Reagents"])
async def update_reagent(reagent_id: int, update: Dict):
    try:
        with get_db() as conn:
            # Update reagent
            conn.execute(
                "UPDATE reagents SET quantity = ?, last_updated = ? WHERE id = ?",
                (update["quantity"], datetime.now().isoformat(), reagent_id)
            )

            # Add history entry
            conn.execute(
                """INSERT INTO reagent_history 
                (reagent_id, user, change, notes) 
                VALUES (?, ?, ?, ?)""",
                (reagent_id, update["user"], update["change"], update.get("notes"))
            )

            return {"status": "success"}
    except Exception as e:
        logging.error(f"Update error: {str(e)}")
        raise HTTPException(500, "Update failed")


# Startup
@app.on_event("startup")
async def startup():
    init_db()
    # Add your Excel import logic here if needed


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)