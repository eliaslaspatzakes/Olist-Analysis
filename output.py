import pandas as pd
from sqlalchemy import create_engine

import os

connection_str = "postgresql+psycopg2://postgres:1131995i%40@localhost:5432/olist"
engine = create_engine(connection_str)

print("Reading data from SQL... (This might take a minute)")

# 2. Run the Query
# We grab the entire 'analytics_master' view we created earlier
query = "SELECT * FROM analytics_master"
df = pd.read_sql(query, engine)

# 3. Save to CSV
output_file = 'olist_dashboard_data.csv'
df.to_csv(output_file, index=False)

print(f"Success! Data exported to '{output_file}'")
print(f"Rows exported: {len(df)}")

print(f"I saved the file here: {os.getcwd()}")