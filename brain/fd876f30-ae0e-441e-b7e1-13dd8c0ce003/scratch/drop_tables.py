import sys
sys.path.append(r"c:\Users\Admin\Downloads\e2ee-trust-simulator\backend")

from app.database import engine, Base
import app.models

print("Dropping all tables from database...")
try:
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped successfully.")
except Exception as e:
    print(f"Error dropping tables: {e}")
