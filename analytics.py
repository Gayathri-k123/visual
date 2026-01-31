import pandas as pd
import os
import glob
from datetime import datetime

# 1. The Math Function (Used for Single Session)
def calculate_engagement(csv_path):
    try:
        df = pd.read_csv(csv_path)
        total_frames = len(df)
        
        if total_frames == 0:
            return 0
            
        # Count Good Behavior (Attentive + Blinking)
        attentive_count = len(df[df['status'] == 'Attentive'])
        blinking_count = len(df[df['status'] == 'Blinking'])
        
        engaged_frames = attentive_count + blinking_count
        score = (engaged_frames / total_frames) * 100
        
        return round(score, 2)
        
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0

# 2. The History Function (Used for Archives)
def get_all_reports():
    reports_data = []
    
    # If folder doesn't exist, return empty list
    if not os.path.exists('reports'):
        return []

    # Find all CSV files in the folder
    files = glob.glob("reports/*.csv")
    
    # Sort them: Newest file first
    files.sort(key=os.path.getmtime, reverse=True)

    for file_path in files:
        try:
            # Extract Date from filename "session_20231015..."
            filename = os.path.basename(file_path)
            time_str = filename.replace('session_', '').replace('.csv', '')
            
            # Make the date look nice
            dt_obj = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            formatted_date = dt_obj.strftime("%b %d, %Y %I:%M %p")

            # Calculate the score for this file
            score = calculate_engagement(file_path)

            # Add to the list
            reports_data.append({
                'date': formatted_date,
                'score': score,
                'filename': filename
            })
        except:
            continue

    return reports_data