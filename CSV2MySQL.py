import csv
import mysql.connector

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="your_database"
)

cursor = db.cursor()

# Read CSV and insert into MySQL
with open('attendance.csv', 'r') as file:
    csv_data = csv.reader(file)
    next(csv_data)  # Skip header if exists
    
    for row in csv_data:
        cursor.execute(
            "INSERT INTO attendance_records (student_id, date, status) VALUES (%s, %s, %s)",
            row
        )

db.commit()
cursor.close()
db.close()
print("Data transfer completed successfully.")