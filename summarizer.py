


# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# model=SentenceTransformer("all-MiniLM-L6-v2")
#
# def load_papers(filename='papers.json'):
#     try:
#         with open(filename,'r') as file:
#             papers=json.load(file)
#         print(f'Loaded {len(papers)} papers from {filename}')
#         return papers
#     except FileNotFoundError:
#         print(f'{filename} not found. returning empty list')
#         return []
#     except json.JSONDecodeError:
#         print(f'Error decoding {filename}')
#         return []
#
# from transformers import pipeline
# summarizer=pipeline('summarization',model='facebook/bart-large-cnn')
#
# def summarize_text(text):
#     input_len=len(text.split())
#     max_len = min(int(0.7 * input_len), 150)   # Dynamically set max_len
#     min_len = max(10, min(50, max_len - 10)) #ensure min_len<max_len
#     summary=summarizer(text,max_length=max_len,min_length=min_len,do_sample=False)
#     return summary[0]['summary_text']
#
#
# def generate_embeddings(papers):
#     paper_texts=[paper['title'] + ' ' + paper['summary'] for paper in papers]
#     embeddings=model.encode(paper_texts,convert_to_numpy=True)
#     return embeddings
#
# #for finding similar files
# def save_faiss_index(papers,faiss_index_path='papers.index'):
#     embeddings = generate_embeddings(papers)
#     dimension = embeddings.shape[1]
#     try:
#         index = faiss.read_index(faiss_index_path)
#     except:
#         index = faiss.IndexFlatL2(dimension) #sets the index such that when searching , it gives faster result based similarity on basis of eucliden dist
#
#     index.add(np.array(embeddings, dtype=np.float32))
#     faiss.write_index(index, faiss_index_path)
#     print(f"Stored {len(papers)} embeddings in FAISS.")
#
# def summarize_papers(papers):
#     summarized_papers=[]
#     for paper in papers:
#         summarized_papers.append({
#             'title': paper['title'],
#             'authors':paper['authors'],
#             'summary':summarize_text(paper['summary']),
#             'id': paper.get('id', ''),
#             'published': paper.get('published', '')
#         })
#     save_faiss_index(summarized_papers)
#     return summarized_papers
#
# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Load papers from JSON
# def load_papers(filename='papers.json'):
#     try:
#         with open(filename, 'r') as file:
#             papers = json.load(file)
#         print(f'Loaded {len(papers)} papers from {filename}')
#         return papers
#     except FileNotFoundError:
#         print(f'{filename} not found. returning empty list')
#         return []
#     except json.JSONDecodeError:
#         print(f'Error decoding {filename}')
#         return []
#
# # Initialize the summarizer pipeline
# from transformers import pipeline
# summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
#
# def summarize_text(text):
#     input_len = len(text.split())
#     max_len = min(int(0.7 * input_len), 150)  # Dynamically set max_len
#     min_len = max(10, min(50, max_len - 10))  # ensure min_len<max_len
#     summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
#     return summary[0]['summary_text']
#
#
# def generate_embeddings(papers):
#     paper_texts = [paper['title'] + ' ' + paper['summary'] for paper in papers]
#     embeddings = model.encode(paper_texts, convert_to_numpy=True)
#     return embeddings
#
# # For finding similar files
# def save_faiss_index(papers, faiss_index_path='papers.index'):
#     embeddings = generate_embeddings(papers)
#     dimension = embeddings.shape[1]
#     try:
#         index = faiss.read_index(faiss_index_path)
#     except:
#         index = faiss.IndexFlatL2(dimension)  # sets the index such that when searching , it gives faster result based similarity on basis of euclidean distance
#
#     index.add(np.array(embeddings, dtype=np.float32))
#
#     faiss.write_index(index, faiss_index_path)
#     print(f"Stored {len(papers)} embeddings in FAISS.")
#
# def summarize_papers(papers):
#     summarized_papers = []
#     for paper in papers:
#         summarized_papers.append({
#             'title': paper['title'],
#             'authors': paper['authors'],
#             'summary': summarize_text(paper['summary']),
#             'id': paper.get('id', ''),
#             'published': paper.get('published', '')
#         })
#         # save_faiss_index(summarized_papers)
#     return summarized_papers

import json
def load_papers(filename='papers.json'):
    try:
        with open(filename,'r') as file:
            papers=json.load(file)
        print(f'Loaded {len(papers)} papers from {filename}')
        return papers
    except FileNotFoundError:
        print(f'{filename} not found. returning empty list')
        return []
    except json.JSONDecodeError:
        print(f'Error decoding {filename}')
        return []

from transformers import pipeline
summarizer=pipeline('summarization',model='facebook/bart-large-cnn')

def summarize_text(text):
    input_len=len(text.split())
    max_len = min(int(0.7 * input_len), 150)   # Dynamically set max_len
    min_len = max(10, min(50, max_len - 10)) #ensure min_len<max_len
    summary=summarizer(text,max_length=max_len,min_length=min_len,do_sample=False)
    return summary[0]['summary_text']

def summarize_papers(papers):
    summarized_papers=[]
    for paper in papers:
        summarized_papers.append({
            'title': paper['title'],
            'authors':paper['authors'],
            'summary':summarize_text(paper['summary']),
            'id': paper.get('id', ''),
            'published': paper.get('published', '')
        })
    return summarized_papers

