from fastapi import FastAPI, HTTPException , Query
import os, json, subprocess
from datetime import datetime
import openai
from typing import Dict
from typing import Optional
from pydantic import BaseModel
import cv2
import sqlite3
import numpy as np
import requests
import markdown
import csv
from bs4 import BeautifulSoup
import duckdb
from git import Repo
from fastapi.responses import JSONResponse
from PIL import Image
import speech_recognition as sr
import glob
import uvicorn
from dotenv import load_dotenv
import base64
import pytesseract


app = FastAPI()
load_dotenv()

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")  

if not AIPROXY_TOKEN:
    print("AIPROXY_TOKEN is NOT set!")
else:
    print("AIPROXY_TOKEN:", AIPROXY_TOKEN)  # Debugging

openai.base_url = "https://aiproxy.sanand.workers.dev/openai/v1/"
openai.api_key = AIPROXY_TOKEN

class TaskRequest(BaseModel):
    task: str
    email: Optional[str] = "user@example.com"
    api_url: Optional[str] = None
    repo_url: Optional[str] = None
    commit_message: Optional[str] = None
    query: Optional[str] = None
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    column: Optional[str] = None
    value: Optional[str] = None
    
# --- Added this line to get the directory of the current script ---
script_dir = os.path.dirname(os.path.abspath(__file__))

TASKS = {
    "install_uv": lambda: install_uv(),
    "format_file": lambda: format_file(os.path.join(script_dir, "data", "format.md")),
    "count_wednesdays": lambda: count_wednesdays(os.path.join(script_dir, "data", "dates.txt")),
    "sort_contacts": lambda: sort_contacts(os.path.join(script_dir, "data", "contacts.json")),
    "extract_recent_logs": lambda: extract_recent_logs(os.path.join(script_dir, "data", "logs/")),
    "generate_markdown_index": lambda: generate_markdown_index(os.path.join(script_dir, "data", "docs/")),
    "extract_email_sender": lambda: extract_email_sender(os.path.join(script_dir, "data", "email.txt")),
    "extract_credit_card": lambda: extract_credit_card(os.path.join(script_dir, "data", "credit_card.png")),
    "find_similar_comments": lambda: find_similar_comments(os.path.join(script_dir, "data", "comments.txt")),
    "calculate_gold_ticket_sales": lambda: calculate_gold_ticket_sales(os.path.join(script_dir, "data", "ticket-sales.db")),
    "fetch_api_data": lambda: fetch_api_data("https://api.example.com/data", os.path.join(script_dir, "data", "api.json")),
    "clone_git_repo": lambda: clone_git_repo("https://github.com/example/repo.git", "Auto-commit"),
    "run_sql_query": lambda: run_sql_query(os.path.join(script_dir, "data", "database.db"), "SELECT COUNT(*) FROM users"),
    "scrape_website": lambda: scrape_website("https://example.com", os.path.join(script_dir, "data", "webpage.txt")),
    "resize_image": lambda: resize_image(os.path.join(script_dir, "data", "image.png"), 300, 300),
    "transcribe_audio": lambda: transcribe_audio(os.path.join(script_dir, "data", "audio.mp3")),
    "convert_md_to_html": lambda: convert_md_to_html(os.path.join(script_dir, "data", "doc.md")),
    "filter_csv": lambda: filter_csv(os.path.join(script_dir, "data", "data.csv"), "status", "active"),
}
"""
@app.post("/run")
async def run_task(request: TaskRequest):
    task_description = request.task.lower()
    function_name = get_function_from_gpt(task_description)

    if function_name == "fetch_api_data":
        return fetch_api_data(request.api_url, os.path.join(script_dir, "data", "api.json"))
    elif function_name == "clone_git_repo":
        return clone_git_repo(request.repo_url, request.commit_message or "Auto-commit")
    elif function_name == "run_sql_query":
        return run_sql_query(os.path.join(script_dir, "data", "database.db"), request.query)
    elif function_name == "scrape_website":
        return scrape_website(request.url, os.path.join(script_dir, "data", "webpage.txt"))
    elif function_name == "resize_image":
        return resize_image(os.path.join(script_dir, "data", "image.png"), request.width, request.height)
    elif function_name == "transcribe_audio":
        return transcribe_audio(os.path.join(script_dir, "data", "audio.mp3"))
    elif function_name == "convert_md_to_html":
        return convert_md_to_html(os.path.join(script_dir, "data", "doc.md"))
    elif function_name == "filter_csv":
        return filter_csv(os.path.join(script_dir, "data", "data.csv"), request.column, request.value)
    else:
        return {"error": f"Unknown task '{function_name}'"}

"""
@app.post("/run")
async def run_task(task: Optional[str] = Query(None), request: Optional[TaskRequest] = None):
    if task:
        task_description = task.lower()  # Handling query param case
    elif request and request.task:
        task_description = request.task.lower()  # Handling JSON body case
    else:
        return {"error": "No task provided"}
    
    function_name = get_function_from_gpt(task_description)

    if function_name == "fetch_api_data":
        return fetch_api_data(request.api_url, os.path.join(script_dir, "data", "api.json")) if request else {"error": "Missing API URL"}
    elif function_name == "clone_git_repo":
        return clone_git_repo(request.repo_url, request.commit_message or "Auto-commit") if request else {"error": "Missing repo URL"}
    elif function_name == "run_sql_query":
        return run_sql_query(os.path.join(script_dir, "data", "database.db"), request.query) if request else {"error": "Missing query"}
    elif function_name == "scrape_website":
        return scrape_website(request.url, os.path.join(script_dir, "data", "webpage.txt")) if request else {"error": "Missing URL"}
    elif function_name == "resize_image":
        return resize_image(os.path.join(script_dir, "data", "image.png"), request.width, request.height) if request else {"error": "Missing dimensions"}
    elif function_name == "transcribe_audio":
        return transcribe_audio(os.path.join(script_dir, "data", "audio.mp3")) if request else {"error": "Missing audio file"}
    elif function_name == "convert_md_to_html":
        return convert_md_to_html(os.path.join(script_dir, "data", "doc.md")) if request else {"error": "Missing markdown file"}
    elif function_name == "filter_csv":
        return filter_csv(os.path.join(script_dir, "data", "data.csv"), request.column, request.value) if request else {"error": "Missing CSV filter parameters"}
    elif function_name == "install_uv":
        return install_uv(request.email)  # Pass email argument
    elif function_name in TASKS:
        return TASKS[function_name]()
    else:
        return {"error": f"Unknown task '{function_name}'"}


    
def get_function_from_gpt(user_input):
    """Uses GPT-4o Mini to determine the correct function to execute."""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You determine the correct function to execute from a list of predefined tasks."},
            {"role": "user", "content": f"User requested: '{user_input}'. Available functions: {list(TASKS.keys())}. Return only the correct function name."}
        ]
    )
    # --- Corrected access to the response content ---
    return response.choices[0].message.content.strip()

@app.get("/read")
async def read_file(path: str):
    """Reads a file and returns its contents."""

    with open(script_dir + path, "r") as file:
        return {"content":file.read()}

#A1

DATA_GEN_URL = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"

def install_uv(email: str):
    try:
        subprocess.run(["pip", "install", "uv"], check=True)
        subprocess.run(["pip", "install", "faker"], check=True)

        # Download datagen.py
        response = requests.get(DATA_GEN_URL)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        datagen_code = response.text

        # Save datagen.py to a temporary file
        datagen_path = os.path.join(script_dir, "temp_datagen.py")
        with open(datagen_path, "w") as f:
            f.write(datagen_code)

        # Execute datagen.py with the email argument
        subprocess.run(["python", datagen_path, email, "--root", script_dir + "/data/"], check=True)


        # Optionally, remove the temporary file
        #os.remove(datagen_path)

        return {"message": "uv installed, datagen.py downloaded and executed"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to install uv or run datagen.py: {e}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to download datagen.py: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

#A2
def format_file(filepath):
    try:
        subprocess.run(["npx", "prettier@3.4.2", "--write", filepath], check=True)
        return {"message": f"Formatted {filepath}"}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Prettier formatting failed")

#A3
DATE_FORMATS = [
    "%Y-%m-%d",          # 2004-10-01
    "%Y/%m/%d %H:%M:%S", # 2014/12/24 17:39:53
    "%Y/%m/%d",          # 2012/04/24
    "%d-%b-%Y",          # 30-Apr-2009
    "%b %d, %Y",         # Jun 19, 2013
    "%Y-%m-%d %H:%M:%S"  # 2018-11-06 10:18:29
]
def parse_date(date_str):
    """Tries multiple date formats and returns a parsed date."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue  # Try next format
    return None  # Return None if no format matches

def count_wednesdays(filepath):
    """Counts the number of Wednesdays in a date file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        raw_dates = [line.strip() for line in f.readlines()]

    valid_dates = [parse_date(date) for date in raw_dates if parse_date(date)]
    wednesday_count = sum(1 for date in valid_dates if date.weekday() == 2)  # Wednesday is 2

    output_path = os.path.join(os.path.dirname(filepath), "dates-wednesdays.txt")
    with open(output_path, "w") as f:
        f.write(str(wednesday_count))
        

    
#A4
def sort_contacts(filepath):
    with open(filepath, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    with open(os.path.join(script_dir, "data", "contacts-sorted.json"), "w") as f:
        f.write(repr(sorted_contacts))
    return None
#A5
def extract_recent_logs(directory):
    logs = sorted(glob.glob(os.path.join(directory, "*.log")), key=os.path.getmtime, reverse=True)[:10]
    with open(os.path.join(script_dir, "data", "logs-recent.txt"), "w") as f:
        for log in logs:
            with open(log, "r") as l:
                f.write(l.readline())
    return {"message": "Recent logs extracted"}

#A6

def generate_markdown_index(base_directory):
    """Generates a JSON index of Markdown files, traversing subdirectories."""
    index = {}  # Initialize the index
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.startswith("# "):
                                index[file_path] = line.strip("# ").strip()
                                break  # Found the heading, move to the next file
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")  # Handle potential file read errors

    # Write all key value pairs of the index in /data/index.json
    output_path = os.path.join(script_dir, "data", "docs", "index.json")  # Save it to script_dir/data, all directory related issues solved!
    try:
        with open(output_path, "w") as f:
            json.dump(index, f, indent=4)  # Ensure proper JSON formatting
    except Exception as e:
        print(f"Error writing to JSON file: {e}")  # Handling json write error!

    return {"message": "Markdown index created"}

#A7
def extract_email_sender(filepath):
    """Extracts the sender's email from an email text file."""
    with open(filepath, "r") as f:
        email_text = f.read()
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract ONLY the sender's email address from this email.  Return ONLY the email address and nothing else:\n{email_text}"}]
    )
    # --- Corrected access to the response content ---
    sender_email = response.choices[0].message.content.strip()
    
    with open(os.path.join(script_dir, "data", "email-sender.txt"), "w") as f:
        f.write(sender_email)
    
    return {"message": f"Extracted sender email: {sender_email}"}

#A8
def extract_text_from_image(filepath):
    """Extract raw text from image using Tesseract OCR."""
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray)  # Extract text from image
    return text.strip()

def extract_credit_card(filepath):
    """Extracts credit card number from an image using OpenCV (OCR)."""
    extracted_text = extract_text_from_image(filepath)

    
    # Use OCR to extract text
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":  f"The following text was extracted from a credit card image: {extracted_text} Identify and extract the credit card number (return only the digits, no spaces or other characters)"}]

        )

    # --- Corrected access to the response content ---
    card_number = response.choices[0].message.content.strip().replace(" ", "")
    
    with open(os.path.join(script_dir, "data", "credit-card.txt"), "w") as f:
        f.write(card_number)
    
    return {"message": "Credit card number extracted"}

#A9
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)  
    return response.data[0].embedding
def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_comments(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]

    # Get embeddings for each comment
    embeddings = [get_embedding(comment) for comment in comments]

    # Find the most similar pair using cosine similarity
    max_sim = -1
    idx1, idx2 = -1, -1
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):  # Avoid duplicate pairs
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_sim:
                max_sim = sim
                idx1, idx2 = i, j

    # Save the most similar comments
    output_path = os.path.join(os.path.dirname(filepath), "comments-similar.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(comments[idx1] + "\n" + comments[idx2])

    return {"message": "Most similar comments found and saved"}


#A10
def calculate_gold_ticket_sales(db_path):
    """Calculates total sales revenue for Gold tickets from an SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0]
    conn.close()
    
    with open(os.path.join(script_dir, "data", "ticket-sales-gold.txt"), "w") as f:
        f.write(str(total_sales))
    
    return {"message": f"Total Gold ticket sales: {total_sales}"}


#B3-B10

def fetch_api_data(api_url, save_path):
    """Fetches data from an API and saves it as a JSON file."""
    response = requests.get(api_url)
    if response.status_code == 200:
        with open(save_path, "w") as f:
            json.dump(response.json(), f, indent=4)
        return {"message": "API data fetched"}
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch API data")


def clone_git_repo(repo_url, commit_message):
    """Clones a Git repository and commits a small change."""
    repo_path = os.path.join(script_dir, "data", "repo")
    Repo.clone_from(repo_url, repo_path)

    file_path = os.path.join(repo_path, "README.md")
    with open(file_path, "a") as f:
        f.write("\n# Auto-generated commit\n")

    repo = Repo(repo_path)
    repo.git.add(update=True)
    repo.index.commit(commit_message)
    repo.remote().push()

    return {"message": "Git repository cloned and updated"}

def run_sql_query(db_path, query):
    """Runs a SQL query on a SQLite or DuckDB database."""
    conn = duckdb.connect(db_path)
    result = conn.execute(query).fetchall()
    conn.close()

    with open(os.path.join(script_dir, "data", "sql_result.txt"), "w") as f:
        f.write(str(result))

    return {"message": "SQL query executed", "result": result}

def scrape_website(url, save_path):
    """Extracts text from a website and saves it."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    with open(save_path, "w") as f:
        f.write(text)

    return {"message": "Website data extracted"}

def resize_image(image_path, width, height):
    """Resizes an image and saves it."""
    image = Image.open(image_path)
    image = image.resize((width, height))
    image.save(image_path)

    return {"message": "Image resized"}

def transcribe_audio(audio_path):
    """Transcribes an audio file to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio)

    with open(os.path.join(script_dir, "data", "audio-transcription.txt"), "w") as f:
        f.write(text)

    return {"message": "Audio transcribed"}

def convert_md_to_html(md_path):
    """Converts a Markdown file to HTML."""
    with open(md_path, "r") as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content)

    with open(os.path.join(script_dir, "data", "output.html"), "w") as f:
        f.write(html_content)

    return {"message": "Markdown converted to HTML"}

def filter_csv(csv_path, column, value):
    """Filters a CSV file and returns matching rows as JSON."""
    result = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row[column] == value:
                result.append(row)

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
