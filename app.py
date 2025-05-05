from flask import Flask, request, jsonify, render_template
from rag_search.vector_db import VectorDatabase
import os
import base64

app = Flask(__name__)

# Initialize global variables
vec = None

@app.route('/')
def index():
    """Serve the frontend."""
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_vector_db():
    """Initialize the VectorDatabase with user-selected models and API keys."""
    data = request.json
    text_model = data.get('text_model')
    image_model = data.get('image_model')
    response_model = data.get('response_model')
    captioning_model = data.get('captioning_model')
    openai_key = data.get('openai_key')
    huggingface_key = data.get('huggingface_key')
    save_dir = data.get('save_dir', None)

    try:
        global vec
        vec = VectorDatabase(
            text_embedding_model=text_model,
            image_embedding_model=image_model,
            response_model=response_model,
            captioning_model=captioning_model,
            openai_api_key=openai_key,
            huggingface_key=huggingface_key,
            save_dir=save_dir
        )
        return jsonify({"message": f"""VectorDatabase initialized successfully with save_dir: {save_dir or 'default ("vector_db")'}!"""})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/embed', methods=['POST'])
def embed():
    """Embed a file or folder."""
    data = request.json
    path = data.get('path')
    is_folder = data.get('is_folder')

    if not vec:
        return jsonify({"error": "VectorDatabase is not initialized. Please select models first."}), 400

    try:
        if is_folder:
            vec.vectorize_folder(folder_path=path)
        else:
            vec.vectorize_file(file_path=path)
        return jsonify({"message": f"Successfully embedded {'folder' if is_folder else 'file'}: {path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/search', methods=['POST'])
def search():
    """Perform a search in the VectorDatabase."""
    data = request.json
    query = data.get('query', {})
    search_location = data.get('search_location', None)

    if not vec:
        return jsonify({"error": "VectorDatabase is not initialized. Please initialize it first."}), 400

    if not query.get('text'):
        return jsonify({"error": "Search query text is required."}), 400

    try:
        # Decode base64 images if provided
        if 'image' in query:
            decoded_images = []
            for image in query['image']: 
                try:
                    decoded_image = base64.b64encode(base64.b64decode(image)).decode("utf-8") 
                    decoded_images.append(decoded_image)
                except Exception as e:
                    return jsonify({"error": f"Invalid image format: {e}"}), 400
            query['image'] = decoded_images

        # Perform the search
        response = vec.run_search(
            search_content=query,
            search_location=search_location
        )
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)