from fastapi import FastAPI, HTTPException, Body
import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime
import traceback
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import uvicorn

app = FastAPI(
    title="Lab Inventory API",
    version="1.0.4",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Persistent database path for Render
DATABASE_NAME = "/var/lib/render/inventory.db"
EXCEL_FILE_NAME = "inventory.xlsx"

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


def initialize_database():
    """Initialize database with tables"""
    try:
        logger.info("Initializing database...")
        os.makedirs("/var/lib/render", exist_ok=True)  # Ensure directory exists

        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

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
            quantity REAL,
            unit TEXT,
            notes TEXT,
            tags TEXT,
            last_updated TEXT
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reagent_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reagent_id INTEGER NOT NULL,
            user TEXT NOT NULL,
            change REAL NOT NULL,
            notes TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(reagent_id) REFERENCES reagents(id)
        )''')

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        log_error(f"Database initialization failed: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the input data"""
    df = df.fillna('')
    required_columns = ['name']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df.replace('', None)


def excel_to_db():
    """Import data from Excel to database"""
    try:
        if not os.path.exists(EXCEL_FILE_NAME):
            logger.warning(f"Excel file {EXCEL_FILE_NAME} not found!")
            return {"warning": "Excel file not found"}

        df = pd.read_excel(EXCEL_FILE_NAME)
        df = clean_data(df)
        df['last_updated'] = datetime.now().isoformat()

        conn = sqlite3.connect(DATABASE_NAME)

        success_count = 0
        error_count = 0
        error_messages = []

        for _, row in df.iterrows():
            try:
                row_dict = {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
                columns = ', '.join(row_dict.keys())
                placeholders = ', '.join(['?'] * len(row_dict))
                sql = f"INSERT OR REPLACE INTO reagents ({columns}) VALUES ({placeholders})"
                conn.execute(sql, tuple(row_dict.values()))
                success_count += 1
            except Exception as e:
                error_count += 1
                error_messages.append(f"Error inserting row {_}: {str(e)}")

        conn.commit()
        conn.close()

        return {
            "total_records": len(df),
            "successful": success_count,
            "failed": error_count,
            "errors": error_messages if error_count > 0 else None
        }
    except Exception as e:
        log_error(f"Excel import failed: {str(e)}")
        return {"error": str(e)}


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


@app.get("/")
async def health_check():
    return {
        "status": "running",
        "app": "Lab Inventory API",
        "version": "1.0.4",
        "time": datetime.now().isoformat()
    }


@app.get("/reagents")
async def get_reagents():
    """Get all reagents"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reagents")
        columns = [column[0] for column in cursor.description]
        return {"reagents": [dict(zip(columns, row)) for row in cursor.fetchall()]}
    except Exception as e:
        log_error(f"Error fetching reagents: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")


@app.put("/reagents/{reagent_id}")
async def update_reagent_quantity(
        reagent_id: int,
        user: str = Body(..., embed=True),
        change: float = Body(..., embed=True),
        notes: Optional[str] = Body(None)
):
    """Update reagent quantity"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Get current quantity (Python 3.7 compatible version)
        cursor.execute("SELECT quantity FROM reagents WHERE id=?", (reagent_id,))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Reagent not found")

        new_quantity = result[0] + change
        if new_quantity < 0:
            raise HTTPException(status_code=400, detail="Quantity cannot be negative")

        # Update reagent
        cursor.execute('''
        UPDATE reagents 
        SET quantity = ?, last_updated = ?
        WHERE id = ?
        ''', (new_quantity, datetime.now().isoformat(), reagent_id))

        # Add history
        cursor.execute('''
        INSERT INTO reagent_history 
        (reagent_id, user, change, notes, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            reagent_id,
            user,
            change,
            notes or f"Quantity adjusted by {change}",
            datetime.now().isoformat()
        ))

        conn.commit()
        return {
            "status": "success",
            "new_quantity": new_quantity,
            "history_id": cursor.lastrowid
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        log_error(f"Update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        conn.close()


@app.get("/reagents/history")
async def get_all_history():
    """Get all history entries"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM reagent_history ORDER BY timestamp DESC')
        columns = [column[0] for column in cursor.description]
        return {"history": [dict(zip(columns, row)) for row in cursor.fetchall()]}
    except Exception as e:
        log_error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        conn.close()


@app.post("/reagents/{reagent_id}/history")
async def add_history_entry(
        reagent_id: int,
        history_entry: Dict[str, Any] = Body(...)
):
    """Add history entry"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Verify reagent exists
        cursor.execute("SELECT id FROM reagents WHERE id=?", (reagent_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Reagent not found")

        # Insert history
        cursor.execute('''
        INSERT INTO reagent_history 
        (reagent_id, user, change, notes, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            reagent_id,
            history_entry.get('user', 'anonymous'),
            history_entry.get('change', 0),
            history_entry.get('notes', ''),
            history_entry.get('timestamp', datetime.now().isoformat())
        ))

        conn.commit()
        return {"status": "success", "id": cursor.lastrowid}
    except Exception as e:
        log_error(f"Error adding history: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        conn.close()


@app.get("/force-import")
async def force_import():
    """Force re-import from Excel"""
    return excel_to_db()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)