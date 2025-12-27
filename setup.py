import pandas as pd
import os
from sqlalchemy import create_engine, text


connection_str = "postgresql+psycopg2://postgres:1131995i%40@localhost:5432/olist"
try:
    engine = create_engine(connection_str)
    with engine.connect() as conn:
        print("Successfully connected to PostgreSQL!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

current_script_folder = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(current_script_folder, 'data')

print(f"Looking for data in: {data_dir}")

if not os.path.exists(data_dir):
    print("ERROR: Still cannot find the data folder.")
    print(f"Please make sure the folder 'data' is inside: {current_script_folder}")
    exit()

files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# --- DATA LOADING ---
for file in files:
    table_name = file.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '')
    print(f"Processing {file} -> Table: {table_name}...")
    
    try:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"   Success! Added {len(df)} rows.")
    except Exception as e:
        print(f"   Error processing {file}: {e}")

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE orders ADD PRIMARY KEY (order_id);"))
        conn.execute(text("ALTER TABLE products ADD PRIMARY KEY (product_id);"))
        conn.execute(text("ALTER TABLE customers ADD PRIMARY KEY (customer_id);"))
        conn.execute(text("ALTER TABLE sellers ADD PRIMARY KEY (seller_id);"))
        conn.commit()
        print("Primary keys added successfully.")
except Exception as e:
    print(f"Note on keys: {e}")

print("\nDone! Your PostgreSQL database is ready.")