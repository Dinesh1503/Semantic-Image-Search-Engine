# import psycopg
# from pathlib import Path
# from dotenv import load_dotenv
# import os
# import numpy as np

# DB_URL = "postgresql://max:pass@localhost:5432/image_db"

# with psycopg.connect(DB_URL) as conn:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 DROP TABLE IF EXISTS images;
#             """)
#             conn.commit()