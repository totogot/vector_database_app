# Vector Database App

## Introduction
This repository provides a Python-based application that allows users to initialize, embed, and search a vector database using text and image embeddings. The intention behind it is to offer an introduction to leveraging LLMs (either run locally or via API endpoints) for the purpose of storing your files in a meaningful format for querying and searching. 

While many providers of vector search functionality require users to have a cloud hosted vector database, or ask you to upload your complete files to their webpages for hosting (which introduces security and privacy concerns), this repository offers a more local approach to achieving some of the functionality. The complete source files remain on our own device at all times, with smaller isolated excerpts of information passed to endpoints for embedding or response generation (or if you have the hardware available, instances of open source models have been integrated into the code to be run locally on your device GPUs).

Lastly, a locally deployed flask application has been built to provide a user-friendly no code interface for users to leverage the solution to search and query their local files.

## Setup
1. Clone the Repository
```unset
git clone https://github.com/totogot/vector_database_app.git
cd vector_database_app
```

2. Create a Virtual Environment
Create a virtual environment to isolate dependencies:
```
python -m venv .venv
```

Activate the virtual environment:

- Windows:
```
.\.venv\Scripts\activate
```

- macOS/Linux:
```
source venv/bin/activate
```

3. Install Requirements
Install the required Python packages:
```
pip install -r requirements.txt
```

## Python Execution
1. Initialize the Vector Database
You can call the main python class from the repo, being sure to pass the correct model arguments supported
```python
from rag_search.vector_db import VectorDatabase

vec = VectorDatabase(
    text_embedding_model = "openai-text-embedding-3-small",
    image_embedding_model = "local-clip-vit-base-patch32",
    response_model = "openai-gpt-4o",
    captioning_model = "openai-gpt-4v",
    openai_api_key = openai_key,
    save_dir = None # assign to default save directory
    )
```

2. Embed Data
Use the embedding functionality to add files or folders to the vector database:
```python
vec.vectorize_folder(folder_path = 'rag_search/data')
```

3. Search the Database
Run a search query against the vector database, using text or images:
```python
with open("./example_graph.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
query = {
    "text": "What drove this trend?",
    "image": [base64_image]
    }

response = vec.run_search(
    search_content = query, 
    search_location = None
)
```

## Running the App
1. Start the Web Application
Run the Flask app to launch the web interface:
```
python app.py
```

2. Access the Web Interface
Open your browser and navigate to:
http://127.0.0.1:5000

3. Use the App
Step 1: Initialize the vector database by entering the required API keys and selecting models.
Step 2: Embed new data or use an existing database (loaded from the "save_dir" path provided).
Step 3: Perform searches using text queries and optional image uploads.