import pandas as pd
import os
import glob
from datetime import datetime

# 1. The Math Function (Used for Single Session & History)
def calculate_engagement(csv_path):
    """
    Calculates the engagement score based on the status logged in the CSV.
    Positive behaviors: Attentive, Blinking, Typing/Reading.
    Negative behaviors: Looking Away, Sleeping, Cheating.
    """
    try:
        # Load the session data
        df = pd.read_csv(csv_path)
        total_frames = len(df)
        
        if total_frames == 0:
            return 0
            
        # --- DEFINING GOOD BEHAVIOR ---
        # We now include 'Typing/Reading' as a positive engagement status
        positive_statuses = ['Attentive', 'Blinking', 'Typing/Reading']
        
        # Count frames where the status matches any of the positive ones
        engaged_frames = len(df[df['status'].isin(positive_statuses)])
        
        # Calculate percentage
        score = (engaged_frames / total_frames) * 100
        
        return round(score, 2)
        
    except Exception as e:
        print(f"Error calculating score for {csv_path}: {e}")
        return 0

# 2. The History Function (Used for Archives/Dashboard)
def get_all_reports():
    """
    Scans the reports folder and returns a list of all session summaries.
    """
    reports_data = []
    
    # Check if folder exists
    if not os.path.exists('reports'):
        return []

    # Find all CSV files
    files = glob.glob("reports/*.csv")
    
    # Sort files by creation time (Newest first)
    files.sort(key=os.path.getmtime, reverse=True)

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            
            # Expected filename format: "session_20231015_123045.csv"
            # Remove 'session_' and '.csv' to get the raw timestamp string
            time_str = filename.replace('session_', '').replace('.csv', '')
            
            # Convert timestamp string to a readable format
            dt_obj = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            formatted_date = dt_obj.strftime("%b %d, %Y %I:%M %p")

            # Calculate the score for this specific file
            score = calculate_engagement(file_path)

            # Append to list for the frontend
            reports_data.append({
                'date': formatted_date,
                'score': score,
                'filename': filename
            })
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")
            continue

    return reports_data

if __name__ == "__main__":
    # Small test script to check if it works independently
    print("Testing Analytics Logic...")
    all_reports = get_all_reports()
    for report in all_reports:
        print(f"Date: {report['date']} | Score: {report['score']}%")