from fastapi import FastAPI, HTTPException
import sqlite3
from fastapi import Body
import pandas as pd
import os
import logging
from datetime import datetime
import traceback
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Lab Inventory API",
    version="1.0.4",
    docs_url="/docs",
    redoc_url="/redoc"
)

DATABASE_NAME = "inventory.db"
EXCEL_FILE_NAME = "inventory.xlsx"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Exposes all headers
)

@app.get("/force-import")
async def force_import():
    return excel_to_db()

@app.on_event("shutdown")
def shutdown_event():
    import os
    os._exit(0)  # Force clean exit

def log_error(message):
    """Log errors with traceback"""
    logger.error(message)
    logger.error(traceback.format_exc())


def initialize_database():
    try:
        logger.info("Initializing database...")
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
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reagent_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reagent_id INTEGER NOT NULL,
            user TEXT NOT NULL,
            change REAL NOT NULL,
            notes TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(reagent_id) REFERENCES reagents(id)
        )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        log_error(f"Database initialization failed: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the input data"""
    # Fill NA/NaN values with empty string
    df = df.fillna('')

    # Ensure required columns exist
    required_columns = ['name']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert empty strings to None for all columns
    df = df.replace('', None)

    return df


def excel_to_db():
    try:
        logger.info(f"Importing data from {EXCEL_FILE_NAME}...")

        if not os.path.exists(EXCEL_FILE_NAME):
            logger.warning(f"Excel file {EXCEL_FILE_NAME} not found!")
            return {"warning": "Excel file not found"}

        df = pd.read_excel(EXCEL_FILE_NAME)
        logger.info(f"Loaded {len(df)} records from Excel")

        # Clean and validate data
        df = clean_data(df)

        # Add timestamp
        df['last_updated'] = datetime.now().isoformat()

        conn = sqlite3.connect(DATABASE_NAME)

        # Insert data in batches to handle errors gracefully
        success_count = 0
        error_count = 0
        error_messages = []

        for _, row in df.iterrows():
            try:
                # Convert row to dictionary and handle None values
                row_dict = {k: (v if pd.notna(v) else None)
                            for k, v in row.to_dict().items()}

                # Prepare SQL and parameters
                columns = ', '.join(row_dict.keys())
                placeholders = ', '.join(['?'] * len(row_dict))
                sql = f"INSERT OR REPLACE INTO reagents ({columns}) VALUES ({placeholders})"

                # Execute insert
                conn.execute(sql, tuple(row_dict.values()))
                success_count += 1
            except Exception as e:
                error_count += 1
                error_msg = f"Error inserting row {_}: {str(e)}"
                error_messages.append(error_msg)
                logger.error(error_msg)

        conn.commit()
        conn.close()

        logger.info(f"Successfully imported {success_count} records, {error_count} failed")

        result = {
            "total_records": len(df),
            "successful": success_count,
            "failed": error_count,
            "errors": error_messages if error_count > 0 else None
        }

        return result
    except Exception as e:
        log_error(f"Excel import failed: {str(e)}")
        return {"error": str(e)}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    try:
        initialize_database()
        result = excel_to_db()
        logger.info(f"Startup result: {result}")

        # If there were errors but some records succeeded, continue running
        if isinstance(result, dict) and result.get('error'):
            if result.get('successful', 0) > 0:
                logger.warning("Application started with partial data")
            else:
                raise RuntimeError("Failed to import any data")
    except Exception as e:
        log_error(f"Startup failed: {str(e)}")
        raise


@app.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "running",
        "app": "Lab Inventory API",
        "version": "1.0.4",
        "time": datetime.now().isoformat()
    }


@app.get("/reagents")
async def get_reagents():
    """Get all reagents with complete data"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reagents")

        # Get column names
        columns = [column[0] for column in cursor.description]

        # Convert to list of dictionaries
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {"reagents": results}
    except Exception as e:
        log_error(f"Error fetching reagents: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/verify-import")
async def verify_import():
    """Verify all data was imported correctly"""
    try:
        # Get database records
        conn = sqlite3.connect(DATABASE_NAME)
        db_df = pd.read_sql("SELECT * FROM reagents", conn)
        conn.close()

        # Get original Excel data
        excel_df = pd.read_excel(EXCEL_FILE_NAME)

        return {
            "database_records": len(db_df),
            "excel_records": len(excel_df),
            "columns_match": set(db_df.columns) == set(excel_df.columns),
            "missing_columns": set(excel_df.columns) - set(db_df.columns),
            "sample_record": db_df.iloc[0].to_dict() if len(db_df) > 0 else None
        }
    except Exception as e:
        log_error(f"Verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Add this endpoint before if __name__ == "__main__":
@app.post("/reagents/{reagent_id}/history")
async def add_history_entry(
        reagent_id: int,
        history_entry: Dict[str, Any] = Body(...)
):
    """Add a history entry for a reagent"""
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
        log_error(f"Error adding history for reagent {reagent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        conn.close()


# Add to main.py (before if __name__ == "__main__":)

@app.get("/reagents/history")
async def get_all_history():
    """Get all history entries for all reagents"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT * FROM reagent_history
        ORDER BY timestamp DESC
        ''')

        columns = [column[0] for column in cursor.description]
        history = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {"history": history}

    except Exception as e:
        log_error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        conn.close()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # Allows access from all network interfaces
        port=8005,
        reload=True
    )


@app.put("/reagents/{reagent_id}")
async def update_reagent_quantity(
        reagent_id: int,
        user: str = Body(..., embed=True),
        change: float = Body(..., embed=True),
        notes: Optional[str] = Body(None)
):
    """Dedicated endpoint for quantity adjustments"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # 1. Get current quantity
        cursor.execute("SELECT quantity FROM reagents WHERE id=?", (reagent_id,))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Reagent not found")

        current_quantity = result[0]
        new_quantity = current_quantity + change

        # 2. Validate new quantity
        if new_quantity < 0:
            raise HTTPException(status_code=400, detail="Quantity cannot be negative")

        # 3. Update database
        cursor.execute('''
        UPDATE reagents 
        SET quantity = ?, last_updated = ?
        WHERE id = ?
        ''', (new_quantity, datetime.now().isoformat(), reagent_id))

        # 4. Record in history
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