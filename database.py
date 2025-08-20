from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse

DB_CONFIG = {
    'host': 'aws-0-ap-southeast-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.aegejyftralxyklrinyn',
    'password': '5sSLogAzvsZquIoL',
    'port': '6543'
}

def get_db_engine():
    encoded_password = urllib.parse.quote(DB_CONFIG['password'])
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(db_url)
    return engine
