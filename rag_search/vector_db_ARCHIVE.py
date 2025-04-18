import pandas as pd
import json
import numpy as np
import fitz
from openai import OpenAI
import base64
import time
import os
import re
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from io import BytesIO
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class VectorDatabase:
    def __init__(
        self, 
        openai_client=None,
        caption_images = False,
        embedding_model = None,
        embedding_processor = None,
        device = None
        ):

        if caption_images and openai_client == None:
            raise ValueError('openai_client must be provided in order to caption images')

        self.openai_client = openai_client
        self.image_captions = caption_images
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = embedding_model.to(self.device)
        self.embedding_processor = embedding_processor
        self.text_data = pd.DataFrame()
        self.image_data = pd.DataFrame()
        self.folder = None
        self.file = None

    @staticmethod
    def clean_json_string(json_string: str) -> str:
        json_string = json_string.strip()
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if match:
            json_string = match.group(0)
        json_string = json_string.replace('```', '')
        
        return json_string.strip()

    @staticmethod
    def normalize_vector(embedding):
        """Normalize embeddings to unit vectors using L2 normalization."""
        norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
        
        return embedding / norm

    def vectorize_folder(self, folder_path, modality_fusing = "concatenate"):
        self.folder = folder_path
        files = [
            os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.endswith((".pdf"))
            ]

        for file in files:
            self.vectorize_file(file, modality_fusing)

        self.text_data.to_pickle(os.path.join(self.folder, "text_data.pkl"))
        self.image_data.to_pickle(os.path.join(self.folder, "image_data.pkl"))  

        return

    def vectorize_file(self, file, modality_fusing = "concatenate"):
        self.fuse_method = modality_fusing
        ext = os.path.splitext(file)[1]

        if ext == ".pdf":
            print("PDF detected")
            self.embed_pdf(file)

        return

    def embed_pdf(self, file):
        doc = fitz.open(file)
        print(f"Processing Doc: {file}")

        for page_num, page in enumerate(doc):

            # Extract text and bounding boxes
            text_blocks = page.get_text("blocks")
            for idx, block in enumerate(text_blocks):
                x0, y0, x1, y1, text = block[0:5]
                if text.strip():
                    txt_entry = pd.DataFrame(
                        [{
                            "document": file,
                            "page_num": page_num,
                            "text_id": f"text_{idx}",
                            "text": text.strip(),
                            "text_vector": self.embed_text(text.strip()),
                            "bbox": [x0, y0, x1, y1]
                        }], 
                        index = [0]
                    )

                    self.text_data = pd.concat(
                        [self.text_data, txt_entry], 
                        ignore_index=True
                    )

            # Extract images and bounding boxes
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base64.b64encode(base_image["image"]).decode("utf-8")

                if self.image_captions:
                    print(f"Captioning Image: pg {page_num}; img {xref}")
                    max_retries = 3
                    for attempt in range(1, max_retries +1): # we will attempt to generate 3 times
                        try:
                            completion = self.caption_image(image_bytes)
                            response = completion.choices[0].message.content
                            caption = json.loads(self.clean_json_string(response))
                            break
                        except Exception as e:
                            print(f"Attempt {attempt} failed for Page {page_num}, Img {xref}: {e}")
                            if attempt < max_retries:
                                time.sleep(1)
                            else:
                                print("All attempts failed - populating with placeholder")
                                caption = {
                                    "Title": "",
                                    "Description": ""
                                    }
                else:
                    caption = {
                        "Title": "",
                        "Description": ""
                    }

                img_name = img[7]
                bbox = page.get_image_bbox(img_name, transform=False) # gets the bbox coordinates

                img_entry = pd.DataFrame(
                    [{
                        "document": file,
                        "page_num": page_num,
                        "image_id": f"image_{xref}",
                        "image_bytes_encoded": base64.b64encode(base_image["image"]).decode("utf-8"),
                        "image_vector": self.embed_image(base_image["image"]),
                        # "image_ext": base_image["ext"],
                        "text_description": caption['Description'].strip(),
                        "text_vector": self.embed_text(caption['Description'].strip()),
                        "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                    }],
                    index = [0]
                )

                self.image_data = pd.concat(
                        [self.image_data, img_entry], 
                        ignore_index=True
                    )

        return

    def caption_image(self, base64_image):
        prompt = """
        can you caption this image with a description?
        provide the response in the following format:
        {{
            "Title": "<image_title>",
            "Description": "<image_description>"
        }}
        """

        caption = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )
        return caption

    def embed_text(self, text):
        """Generate CLIP embedding for a text input."""
        inputs = self.embedding_processor(
            text=text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model.get_text_features(**inputs)

        return self.normalize_vector(embedding).cpu().numpy()

    def embed_image(self, image_bytes):
        """Generate CLIP embedding for an image from bytes."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self.embedding_processor(
            images=image, return_tensors="pt"
            ).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model.get_image_features(**inputs)

        return self.normalize_vector(embedding).cpu().numpy()

    def fuse_embeddings(text_emb, image_emb):
        """Fuse text and image embeddings using specified method."""
        if self.fuse_method == "average":
            fused_emb = (text_emb + image_emb) / 2
        elif self.fuse_method == "concatenate":
            fused_emb = np.concatenate([text_emb, image_emb], axis=-1)
        else:
            raise ValueError("Invalid method. Use 'average' or 'concatenate'.")
        
        return fused_emb

    def search_answer(
        self,
        question,
        search_folder = None,
        search_file = None,
        mode = 'text',
        top_n = 5
        ):

        if search_folder and search_file:
            raise ValueError("both folder and file specified for search - please only assign one")

        if not self.folder:
            if not search_folder and not search_file:
                raise ValueError("no default folder assigned, and search folder or file is not specified")
            elif search_folder:
                print(f"search folder specified: {search_folder}")
                self.search_loc = search_folder
                self.search_granularity = "folder"
            else: 
                print(f"search file specified: {search_file}")
                self.search_loc = search_file
                self.search_granularity = "file"
        elif self.folder:
            if not search_folder and not search_file:
                print(f"no search folder or file specified, defaulting to assigned folder {self.folder}")
            elif search_folder:
                print(f"search folder specified: {search_folder}, using instead of default assigned folder {self.folder}")
                self.search_loc = search_folder
                self.search_granularity = "folder"
            else: 
                print(f"search file specified: {search_file}, using instead of default assigned folder {self.folder}")
                self.search_loc = search_file
                self.search_granularity = "file"
        
        if mode in ("text", "image", "text_image"):
            self.search_mode = mode
        else:
            raise ValueError("only modes 'text', 'image' or 'text_image' currently supported")
        
        search_vectors = self.compile_search_range()
        embed_question = self.embed_text(question)

        relevant_references = self.return_similar(embed_question, search_vectors, top_n)

        return relevant_references.drop(columns=["vector"])

    
    def compile_search_range(self):
        search_vectors = pd.DataFrame()

        fol_path = Path(self.search_loc)
        fol_path = fol_path if fol_path.is_dir() else fol_path.parent

        print(f"Searching for {self.search_mode} embeddings")
        if self.search_mode == "text":
            # load the text file
            vector_file = pd.read_pickle(os.path.join(fol_path, "text_data.pkl"))

            # cut if we are only interested in a single file
            if self.search_granularity == "file":
                vector_file = vector_file[vector_file['document']==self.search_loc]

            search_vectors = pd.concat([
                search_vectors, vector_file[['document', 'page_num','text_vector', 'text']]
                ], ignore_index=True
                )
            
            # try the image file
            vector_file = pd.read_pickle(os.path.join(fol_path, "image_data.pkl"))

            # cut if we are only interested in a single file
            if self.search_granularity == "file":
                vector_file = vector_file[vector_file['document']==self.search_loc]

            if "text_vector" in vector_file.columns:
                search_vectors = pd.concat([
                search_vectors, vector_file[['document', 'page_num','text_vector', 'text_description']].rename(columns={"text_description":"text"})
                ], ignore_index=True
                )

            search_vectors = search_vectors.rename(columns={"text_vector": "vector"})
        
        elif self.search_mode == "image":
            # load the image file
            vector_file = pd.read_pickle(os.path.join(fol_path, "image_data.pkl"))

            # cut if we are only interested in a single file
            if self.search_granularity == "file":
                vector_file = vector_file[vector_file['document']==self.search_loc]

            search_vectors = pd.concat([
                search_vectors, vector_file[['document', 'page_num','image_vector', 'text_description']].rename(columns={"text_description":"text"})
                ], ignore_index=True
                )

            search_vectors = search_vectors.rename(columns={"image_vector": "vector"})

        return search_vectors

    def return_similar(self, question_vector, reference_vectors, top_n):

        max_length = max([vec.shape[1] for vec in reference_vectors['vector']])  # Taking the first dimension
        padded_question = self.pad_embedding(question_vector, max_length)

        padded_references = [self.pad_embedding(vec, max_length) for vec in reference_vectors['vector']]
        padded_references = np.array(padded_references).reshape(len(padded_references), -1)

        similarities = cosine_similarity(padded_question, padded_references)
        top_indices = similarities.argsort()[0][-top_n:][::-1]
        print(f"Top Similarity Scores: {np.sort(similarities[0])[-top_n:][::-1]}")

        return reference_vectors.iloc[top_indices]


    
    @staticmethod
    def pad_embedding(embedding, target_length):
        """Pad embeddings to the target length with zeros."""
        padding_length = target_length - embedding.shape[1]
        if padding_length > 0:
            padding = np.zeros((embedding.shape[0], padding_length))
            padded_embedding = np.concatenate([embedding, padding], axis=1)
        else:
            padded_embedding = embedding[:, :target_length]
        return padded_embedding

            

                
                









####### RUN

#### LOAD KEY
with open("../keys/mvp_projects_key.txt","r") as f:
    api_key = f.read()
client_general = OpenAI(api_key=api_key)

#### INITIATE VECTOR CLASS
vec = VectorDatabase(
    openai_client = client_general,
    caption_images = True,
    embedding_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    embedding_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    )

#### VECTORIZE ALL FILES IN FOLDER
# vec.vectorize_folder('./rag_search/data')

#### SEARCH FOR RESPONSE
question = "How has Hebbia's revenue grown in recent years?"
folder = "./rag_search/data"
response = vec.search_answer(question, folder, mode = "image", top_n = 5)
response.to_csv("./test_response.csv")