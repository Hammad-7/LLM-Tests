import re
import json 
import csv
from datetime import datetime, timedelta

CHAT_PATTERN = re.compile(r'(\d{2}/\d{2}/\d{4}), (\d{1,2}:\d{2}\s?[ap]m) - ([^:]+): (.+)')

def parse_chat(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
    
    
    messages = []
    last_match = None
    
    for line in lines:
        if "<Media omitted>" in line:  # Skip media messages
            continue

        match = CHAT_PATTERN.match(line)
        if match:
            if last_match:
                messages.append(last_match)
            date_str, time_str, sender, message = match.groups()
            try:
                time_str = time_str.replace("\u202f", " ")  # Fix non-breaking space issue
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %I:%M %p")
            except ValueError:
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            last_match = {"timestamp": timestamp, "sender": sender, "message": message}
        elif last_match:
            last_match["message"] += f" {line}"  # Append to the previous message
    
    if last_match:
        messages.append(last_match)
    
    return messages

# Function to preprocess messages
def preprocess_messages(messages, user_name, context_window=5, time_threshold=24):
    preprocessed_data = []
    context = []
    last_timestamp = None
    last_sender = None
    
    for msg in messages:
        timestamp, sender, message = msg["timestamp"], msg["sender"], msg["message"]
        
        # Flush context if time gap exceeds threshold
        if last_timestamp and (timestamp - last_timestamp > timedelta(hours=time_threshold)):
            context = []
        
        # Club consecutive messages from the same sender
        if last_sender == sender and preprocessed_data:
            preprocessed_data[-1]["message"] += f" {message}"
        else:
            preprocessed_data.append({"timestamp": timestamp, "sender": sender, "message": message})
        
        last_timestamp = timestamp
        last_sender = sender
    
    # Structure data into input-output pairs
    structured_data = []
    context = []
    
    for i in range(len(preprocessed_data)):
        msg = preprocessed_data[i]
        if msg["sender"] != user_name:
            j = i + 1
            response = ""
            while j < len(preprocessed_data) and preprocessed_data[j]["sender"] == user_name:
                response += " " + preprocessed_data[j]["message"]
                j += 1
            
            if response:
                structured_data.append({
                    "input": msg["message"],
                    "response": response.strip(),
                    "context": context[-context_window:] 
                })
                context.append({"input": msg["message"], "response": response.strip()})
                
                if j < len(preprocessed_data) and (preprocessed_data[j]["timestamp"] - msg["timestamp"] > timedelta(hours=time_threshold)):
                    context = []
    
    return structured_data

def save_to_csv(data, filename):
    # Open the file for writing and create a CSV writer object
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["input", "response", "context"])
        writer.writeheader()
        for entry in data:
            writer.writerow({
                "input": entry["input"],
                "response": entry["response"],
                "context": " ".join([f'{item["input"]} -> {item["response"]}' for item in entry["context"]])
            })

def preprocess_chats(username:str, file_path):
    user_name = username  # Change this to your actual WhatsApp name
    raw_messages = parse_chat(file_path=file_path)
    structured_data = preprocess_messages(raw_messages, user_name)

    return structured_data

if __name__ == "__main__":
    # user_name = "Hammad"  # Change this to your actual WhatsApp name
    # raw_messages = parse_chat("whatsapp_chats.txt")
    structured_data = preprocess_chats(username="Hammad", file_path="whatsapp_chats.txt")
    
    with open("preprocessed_chat.json", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=4, ensure_ascii=False)

    save_to_csv(structured_data, "preprocessed_chat.csv")
    
    print("Preprocessing complete. Data saved to preprocessed_chat.json")