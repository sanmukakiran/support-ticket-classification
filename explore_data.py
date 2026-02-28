import pandas as pd

df = pd.read_csv("customer_support_tickets.csv")

with open("data_summary.txt", "w") as f:
    f.write(f"Dataset Shape: {df.shape}\n\n")
    f.write("Columns:\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    
    f.write("\nFirst 3 rows:\n")
    f.write(df.head(3).to_string())
    
    f.write("\n\nTicket Priority Counts:\n")
    if 'Ticket Priority' in df.columns:
        f.write(df['Ticket Priority'].value_counts().to_string())
        
    f.write("\n\nTicket Type Counts:\n")
    if 'Ticket Type' in df.columns:
        f.write(df['Ticket Type'].value_counts().to_string())
