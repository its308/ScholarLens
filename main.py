# import json
# import requests
# import xml.etree.ElementTree as ET #for parsing xml data
# import faiss
# from summarizer import load_papers,summarize_papers,save_faiss_index
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import torch
# torch.set_num_threads(1)
# def fetch_arxiv_papers(query):
#     url=f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10"
#     response=requests.get(url)
#
#     if response.status_code==200:
#         print("Papers are fetched Successfully")
#         return response.text # just get raw file as of now
#     else:
#         print(f"Error in Fetching the papers :{response.status_code}")
#         return None
#
#
#
# def parse_xml(data):
#     tree=ET.ElementTree(ET.fromstring(data))
#     root=tree.getroot()
#
#
#     papers=[]
#
#     for entry in root.findall('{http://www.w3.org/2005/Atom}entry'): #python break the string by ':' and then connect 0th index with passed namespace
#         title=entry.find('{http://www.w3.org/2005/Atom}title').text
#         authors=[author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
#         summary=entry.find('{http://www.w3.org/2005/Atom}summary').text
#         paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text
#         published = entry.find('{http://www.w3.org/2005/Atom}updated').text
#
#         papers.append({
#             'title':title,
#             'authors':authors,
#             'summary':summary,
#             'id':paper_id,
#             'published':published
#         })
#     return papers
#
# def save_papers_to_json(papers,filename='papers.json'):
#     try:
#         with open(filename,'r') as file:
#             try:
#                 existing_papers=json.load(file)
#             except json.JSONDecodeError:
#                 existing_papers = []
#     except FileNotFoundError:
#         existing_papers=[]
#
#  #checking here that any papers are not duplicates i.e. getting repeated
#     unique_papers=[]
#     existing_titles={paper.get('title','').strip().lower() for paper in existing_papers}
#     for paper in papers:
#         title = paper.get('title', '').strip().lower()
#         if title and title not in existing_titles:
#             unique_papers.append(paper)
#             existing_titles.add(title)
#
#
#     if unique_papers:
#         existing_papers.extend(unique_papers)
#         with open(filename,'w') as file:
#             json.dump(existing_papers,file,indent=4)
#         print(f"Saved Papers to {filename} ")
#     else:
#         print('No new unique papers to save')
#
#
#
#
# def search_papers(query, papers):
#     results = []
#     for paper in papers:
#         if query.lower() in paper['title'].lower() or query.lower() in paper['summary'].lower():
#             results.append(paper)
#     return results
#
# def search_papers_by_embeddings(query,faiss_index_path='papers.index'):
#     index=faiss.read_index(faiss_index_path)
#     model=SentenceTransformer("all-MiniLM-L6-v2")
#     query_embd=model.encode(query,convert_to_numpy=True).astype(np.float32)
#
# # D-distnaces(similarity) b/w papers and query
#     D,I=index.search(query_embd,k=5) #to return top 5 papers matches
#     return I[0]
#
#
# def reset_papers_json(filename='papers.json'):
#     with open(filename,'w') as file:
#         file.write('[]')
#         print(f'Reset {filename} to an empty list')
#
# if __name__=='__main__':
#     reset_papers_json() #it will clear previous papers
#     query='Deep Learning'
#     data=fetch_arxiv_papers(query)
#     if data:
#         papers=parse_xml(data)
#         if not papers:
#             print("No papers found or parsing error.")
#         else:
#             print(f"Found {len(papers)} Papers :")
#             for i,paper in enumerate(papers):
#                 print(f'\nFor Paper {i+1} :')
#                 print(f"\nTitle : {paper['title']}")
#                 print(f"Authors: {', '.join(paper['authors'])}")
#                 print(f"Summary: {paper['summary'][:200]}")
#         save_papers_to_json(papers)
#
#         search_query=input('Enter search query :')
#         indices=search_papers(search_query,papers)
#
#         if indices:
#             print(f"\nFound {len(indices)} matching Papers :")
#             for i in indices:
#                 if i in indices:
#                     paper=papers[i]
#                     print(f"\nTitle : {paper['title']}")
#                     print(f"Authors: {', '.join(paper['authors'])}")
#                     print(f"Summary: {paper['summary'][:200]}")
#         else:
#             print('No papers found matching query')
#
# papers=load_papers()
# if papers:
#     summarized_papers=summarize_papers(papers)
#     save_faiss_index(summarized_papers)  # Save embeddings to FAISS
#     print('\n-------SUMMARIZED PAPERS ------')
#     for paper in summarized_papers:
#         print(f'\n Title of paper : {paper['title']}')
#         print(f' Summary of paper : {paper['summary']}')
#
# import json
# import requests
# import xml.etree.ElementTree as ET  # for parsing xml data
# import faiss
# from summarizer import load_papers, summarize_papers, save_faiss_index
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import torch  # Added for controlling threading behavior
#
# # Set PyTorch to use only one thread (disable parallelism)
# # torch.set_num_threads(1)
#
# def fetch_arxiv_papers(query):
#     url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10"
#     response = requests.get(url)
#
#     if response.status_code == 200:
#         print("Papers are fetched Successfully")
#         return response.text  # just get raw file as of now
#     else:
#         print(f"Error in Fetching the papers :{response.status_code}")
#         return None
#
#
# def parse_xml(data):
#     tree = ET.ElementTree(ET.fromstring(data))
#     root = tree.getroot()
#
#     papers = []
#
#     for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):  # python break the string by ':' and then connect 0th index with passed namespace
#         title = entry.find('{http://www.w3.org/2005/Atom}title').text
#         authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
#         summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
#         paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text
#         published = entry.find('{http://www.w3.org/2005/Atom}updated').text
#
#         papers.append({
#             'title': title,
#             'authors': authors,
#             'summary': summary,
#             'id': paper_id,
#             'published': published
#         })
#     return papers
#
#
# def save_papers_to_json(papers, filename='papers.json'):
#     try:
#         with open(filename, 'r') as file:
#             try:
#                 existing_papers = json.load(file)
#             except json.JSONDecodeError:
#                 existing_papers = []
#     except FileNotFoundError:
#         existing_papers = []
#
#     # checking here that any papers are not duplicates i.e. getting repeated
#     unique_papers = []
#     existing_titles = {paper.get('title', '').strip().lower() for paper in existing_papers}
#     for paper in papers:
#         title = paper.get('title', '').strip().lower()
#         if title and title not in existing_titles:
#             unique_papers.append(paper)
#             existing_titles.add(title)
#
#     if unique_papers:
#         existing_papers.extend(unique_papers)
#         with open(filename, 'w') as file:
#             json.dump(existing_papers, file, indent=4)
#         print(f"Saved Papers to {filename} ")
#     else:
#         print('No new unique papers to save')
#
#
# def search_papers(query, papers):
#     results = []
#     for paper in papers:
#         if query.lower() in paper['title'].lower() or query.lower() in paper['summary'].lower():
#             results.append(paper)
#     return results
#
#
# def search_papers_by_embeddings(query, faiss_index_path='papers.index'):
#     index = faiss.read_index(faiss_index_path)
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     query_embd = model.encode(query, convert_to_numpy=True).astype(np.float32)
#
#     # D-distances(similarity) b/w papers and query
#     D, I = index.search(query_embd, k=5)  # to return top 5 papers matches
#     return I[0]
#
#
# def reset_papers_json(filename='papers.json'):
#     with open(filename, 'w') as file:
#         file.write('[]')
#         print(f'Reset {filename} to an empty list')
#
#
# if __name__ == '__main__':
#     reset_papers_json()  # it will clear previous papers
#     query = 'Deep Learning'
#     data = fetch_arxiv_papers(query)
#     if data:
#         papers = parse_xml(data)
#         print(f"Found {len(papers)} Papers :")
#         for i, paper in enumerate(papers):
#             print(f'\nFor Paper {i + 1} :')
#             print(f"\nTitle : {paper['title']}")
#             print(f"Authors: {', '.join(paper['authors'])}")
#             print(f"Summary: {paper['summary'][:200]}")
#         save_papers_to_json(papers)
#
#         # search_query = input('Enter search query :')
#         # indices = search_papers(search_query, papers)
#         #
#         # if indices:
#         #     print(f"\nFound {len(indices)} matching Papers :")
#         #     for i in indices:
#         #         if i in indices:
#         #             print(f"\nTitle : {paper['title']}")
#         #             print(f"Authors: {', '.join(paper['authors'])}")
#         #             print(f"Summary: {paper['summary'][:200]}")
#         # else:
#         #     print('No papers found matching query')
#
# papers = load_papers()
# if papers:
#     summarized_papers = summarize_papers(papers)
#     # save_faiss_index(summarized_papers)  # Save embeddings to FAISS
#     print('\n-------SUMMARIZED PAPERS ------')
#     for paper in summarized_papers:
#         print(f'\n Title of paper : {paper["title"]}')
#         print(f' Summary of paper : {paper["summary"]}')


import json
from pathlib import Path

import nltk
import requests
import xml.etree.ElementTree as ET #for parsing xml data

import torch.backends.mps
import torch
from summarizer import load_papers,summarize_papers
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset, load_from_disk
import re
import google.generativeai as genai
from transformers import pipeline
from nltk.corpus import stopwords
# summarizer=pipeline('summarization',model='facebook/bart-large-cnn')

# Google_api_key='AIzaSyCFjHwUdOGdFSCH7fKXFz4Ljr3i9df0piA'
# genai.configure(api_key=Google_api_key)
# gemini_model=genai.GenerativeModel('gemini-2.0-flash')

def load_nltk_resources():
    nltk.download(stopwords)
    return stopwords.words('english')
# token='---Removed from here--'
# login(token)

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove any non-ASCII characters
    text = ''.join(c for c in text if ord(c) < 128)
    return text

def extract_keywords(text,n=5):
    from collections import defaultdict
    import nltk
    nltk.download('stopwords',quiet=True)
    from nltk.corpus import stopwords
    text=text.lower()
    word_counts=defaultdict(int)
    for word in text.split():
        if word.isalpha() and word not in stopwords.words('english'):
            word_counts[word]+=1
    sorted_words=sorted(word_counts.items(),key=lambda x:-x[1])[:n]  # -x[1] denotes that sort on basis of wordcount value
    return [word for word,_ in sorted_words] #return top n keywords(most frequent)


def setup_device():
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        torch.mps.empty_cache()
    print(f'Device is set to {device}')
    return device

# tokenizer=RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
# retriever = RagRetriever.from_pretrained(
#     "facebook/rag-sequence-nq",
#     index_name="custom",
#     use_dummy_dataset=True
# )
# rag_model=RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

def initialize_models():

    device=setup_device()
    summarizer=pipeline('summarization', model='facebook/bart-large-cnn')
    model_name='google/flan-t5-base'
    generator=T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer=T5Tokenizer.from_pretrained(model_name)
    embd_model=SentenceTransformer('all-MiniLM-L6-v2')

    return summarizer, generator, tokenizer, embd_model, device

def setup_gemini(api_key='AIzaSyCFjHwUdOGdFSCH7fKXFz4Ljr3i9df0piA'):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')
def fetch_arxiv_papers(query):
    url=f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=20"
    response=requests.get(url)

    if response.status_code==200:
        print("Papers are fetched Successfully")
        return response.text # just get raw file as of now
    else:
        print(f"Error in Fetching the papers :{response.status_code}")
        return None

def parse_xml(data):
    tree=ET.ElementTree(ET.fromstring(data))
    root=tree.getroot()


    papers=[]

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'): #python break the string by ':' and then connect 0th index with passed namespace
        title=entry.find('{http://www.w3.org/2005/Atom}title').text
        authors=[author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        summary=entry.find('{http://www.w3.org/2005/Atom}summary').text
        paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text
        published = entry.find('{http://www.w3.org/2005/Atom}updated').text

        papers.append({
            'title':title,
            'authors':authors,
            'summary':summary,
            'id':paper_id,
            'published':published
        })
    return papers

def save_papers_to_json(papers,filename='papers.json'):
    try:
        with open(filename,'r') as file:
            try:
                existing_papers=json.load(file)
            except json.JSONDecodeError:
                existing_papers = []
    except FileNotFoundError:
        existing_papers=[]

 #checking here that any papers are not duplicates i.e. getting repeated
    unique_papers=[]
    existing_titles={paper.get('title','').strip().lower() for paper in existing_papers}
    for paper in papers:
        title = paper.get('title', '').strip().lower()
        if title and title not in existing_titles:
            unique_papers.append(paper)
            existing_titles.add(title)


    if unique_papers:
        existing_papers.extend(unique_papers)
        with open(filename,'w') as file:
            json.dump(existing_papers,file,indent=4)
        print(f"Saved Papers to {filename} ")
    else:
        print('No new unique papers to save')
    return existing_papers

def create_faiss_index(papers,model):
    embeddings=[]
    for paper in papers:
        embeddings.append(model.encode(paper['summary'],convert_to_numpy=True))
    embeddings=np.stack(embeddings)
    index=faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index,embeddings

def search_faiss(query,index,papers,model):
    query_embd=model.encode([query],convert_to_numpy=True)
    distances,indices=index.search(query_embd,k=5)
    results=[]
    for i in indices[0]:
        results.append(papers[i])
    return results


def search_papers(query, papers):
    results = []
    for paper in papers:
        if query.lower() in paper['title'].lower() or query.lower() in paper['summary'].lower():
            results.append(paper)
    return results

def reset_papers(filename='papers.json'):
    with open(filename,'w') as file:
        file.write('[]')
        print(f'Reset {filename} to an empty list')

# def search_with_rag(query,papers,index):
#     results=search_faiss(query,index,papers,model) # retriving top 5 most relevent paper as per query with faiss
#     # inputs
#     input_text=[paper['summary'] for paper in results]
#     #input encoder
#     inputs=tokenizer(input_text,padding=True,truncation=True,return_tensors='pt',max_length=512) #pt means pytorch type
#     #outputs
#     outputs=rag_model.generate(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],num_beams=3,num_return_sequences=1) #num_beams decides how many next words/sentences are considered at each step
#     #output decoder
#     generated_text=tokenizer.decode(outputs[0],skip_special_tokens=True)
#
#     return generated_text

def convert_to_hf_dataset(papers,embd_model):
        return Dataset.from_dict({
            'title':[p['title'] for p in papers],
            'text': [p['summary'] for p in papers],
            'embeddings':[embd_model.encode(p['summary']) for p in papers]
        })
# def clean_text(text):
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
#     # Remove any non-ASCII characters
#     text = ''.join(c for c in text if ord(c) < 128)
#     return text

def save_faiss_index(index,path='faiss_index.index'):
    faiss.write_index(index,path)

def load_faiss_index(path='faiss_index.index'):
    return faiss.read_index(path)

def prepare_dataset_dir(papers,embd_model):
    hf_dataset=convert_to_hf_dataset(papers,embd_model)
    hf_dataset.save_to_disk('papers_hf')
    hf_dataset=hf_dataset.train_test_split(test_size=0.2)
    # Create required subdirectory structure
    # (Path("papers_hf") / "data").mkdir(exist_ok=True)
    hf_dataset.save_to_disk('paper_hf')
    return hf_dataset

def general_search_with_rag(query,papers,generator,tokenizer,device):

    # context=' '.join([f'Paper:{p['title']} -{p['summary']}' for p in papers])
    # formatted_input=f'Answer this question based on the scientific papers: {query}\n\nContext from papers: {context}'
    paper_contexts=[]
    for i, paper in enumerate(papers):
        clean_title = clean_text(paper['title'])
        clean_summary = clean_text(paper['summary'])
        context = f"[{i+1}] {clean_title}\n{clean_summary}"
        paper_contexts.append(context)

        # Join with clear separation
    full_context = "\n\n".join(paper_contexts)
    prompt = f"question: {query} context: {full_context}"

    # More directive prompt
    # formatted_input = f"""Based on the following research papers, please answer this question: "{query}"

    # PAPERS:
    # {context}
    #
    # Answer the question directly using information from these papers. Provide a specific, informative response."""
# Generate response
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)

    # inputs=tokenizer(prompt,return_tensors='pt',max_length=1024,truncation=True).to(device)
    outputs=generator.generate(input_ids=input_ids,max_length=150, min_length=50,
                               num_beams=5,do_sample=False,
        no_repeat_ngram_size=3,early_stopping=True)

    response=tokenizer.decode(outputs[0],skip_special_tokens=True)
    response = clean_text(response)
    return response

def gemini_general_query(query,topic,model,papers=None,generator=None,tokenizer=None,device=None):
    try:
        prompt=f"""Topic : {topic}
                    Question : {query}
Provide a detailed, specific answer to this question based on current scientific understanding. Include relevant facts and concepts."""
        response=model.generate_content(prompt)
        return response.text,[]
    except Exception as e:
        print(f'Error with Gemini API :{e}')
        print('giving custom RAG Response...')
        return general_search_with_rag(query,papers,generator,tokenizer,device)

# def extract_keywords(text,n=5):
#     from collections import defaultdict
#     import nltk
#     nltk.download('stopwords',quiet=True)
#     from nltk.corpus import stopwords
#     text=text.lower()
#     word_counts=defaultdict(int)
#     for word in text.split():
#         if word.isalpha() and word not in stopwords.words('english'):
#             word_counts[word]+=1
#     sorted_words=sorted(word_counts.items(),key=lambda x:-x[1])[:n]  # -x[1] denotes that sort on basis of wordcount value
#     return [word for word,_ in sorted_words] #return top n keywords(most frequent)



def specific_search_with_rag(paper_num,query,papers,generator,tokenizer,device):
    if 0<=paper_num<len(papers):
        paper=papers[paper_num]
        paper_context={
        'title' : clean_text(paper['title']),
        'authors' : clean_text(', '.join(paper['authors'])),
        'summary': clean_text(paper['summary']),
        'keywords':extract_keywords(paper['summary'])
        }

        prompt = f"""Answer this question about the paper titled "{paper_context['title']}": {query}

        CONTEXT:
        {paper_context['summary']}

        Generate a concise response focusing only on information available in this paper. If the paper's abstract doesn't address this question, simply state that the information isn't provided in the abstract."""


        # formatted_input = fBased on this research paper, please answer this question: "{query}"
    #
    #     PAPER DETAILS:
    #     TITLE: {paper['title']}
    #     AUTHORS: {', '.join(paper['authors'])}
    #     PUBLICATION DATE: {paper['published']}
    #     ABSTRACT: {paper['summary']}
    #
    #     Answer the question directly using only information from this paper. Provide a specific, informative response.

        input_ids = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)

        # inputs=tokenizer(prompt,return_tensors='pt',max_length=1024,truncation=True).to(device)
        outputs = generator.generate(input_ids=input_ids, max_length=300, min_length=50,
                                     num_beams=5, do_sample=False,
                                     no_repeat_ngram_size=3, early_stopping=True)
        response=tokenizer.decode(outputs[0],skip_special_tokens=True)
        return response,[paper]
    else:
        return f'Error : Paper number {paper_num+1} is not found. Please select a no. b/w 1 and {len(papers)}',[]

def generate_summary(text,summarizer):
    input_len = len(text.split())
    max_len = min(int(0.7 * input_len), 150)
    min_len = max(10, min(50, max_len - 10))
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

def display_papers(papers):
    print(f'\n top 5 most relevent papers for your query:')
    for i, paper in enumerate(papers):
        print(f"{i + 1}. Title: {paper['title']}")
        print(f"      Summary: {generate_summary(paper['summary'])}")




'''
if __name__=='__main__':
    # reset_papers_json() #it will clear previous papers

    Path("papers_hf").mkdir(exist_ok=True)
    reset_papers_json()
    main_query=input('Enter your main research topic: ')
    data=fetch_arxiv_papers(main_query)
    if data:
        all_papers=parse_xml(data)

        # print(f"Found {len(papers)} Papers :")
        # for i,paper in enumerate(papers):
        #     print(f'\nFor Paper {i+1} :')
        #     print(f"\nTitle : {paper['title']}")
        #     print(f"Authors: {', '.join(paper['authors'])}")
        #     print(f"Summary: {paper['summary'][:200]}")

        save_papers_to_json(all_papers)


        # create faiss index
        embd_model = SentenceTransformer('all-MiniLM-L6-v2')

        # hf_dataset=convert_to_hf_dataset(all_papers,embd_model)
        # hf_dataset.save_to_disk('papers_hf')

        all_index, _ = create_faiss_index(all_papers,embd_model)
        save_faiss_index(all_index)
        # will retrieve top 5 most relevent papers
        relevent_papers=search_faiss(main_query,all_index,all_papers,embd_model)
        hf_dataset=convert_to_hf_dataset(relevent_papers,embd_model)
        hf_dataset.save_to_disk('papers_hf')

        print(f'\n top 5 most relevent papers for your query:')
        for i,paper in enumerate(relevent_papers):
            print(f'{i+1}.Title :{paper['title']}')
            print(f'      Summary:{generate_summary(paper['summary'])}')

        results=search_papers(search_query,papers)

        results=search_faiss(search_query,index,papers,model)

        # if results:
        #     print(f"\nFound {len(results)} matching Papers :")
        #     for i, paper in enumerate(results):
        #         print(f"\nTitle : {paper['title']}")
        #         print(f"Authors: {', '.join(paper['authors'])}")
        #         print(f"Summary: {paper['summary'][:200]}")
        # else:
        #     print('No papers found matching query')
        # rag_result=search_with_rag(search_query,papers,index)
        # print(f'RAG Model Response :{rag_result}')



        # converting papers to huggingface dataset
        # hf_dataset=convert_to_hf_dataset(papers)


        RAG-----------> # did'nt worked as it was giving very poor and vague responses
        # tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')
        # retriever = RagRetriever.from_pretrained(
        #     "facebook/rag-sequence-nq",
        #     index_name="custom",
        #     passages_path="papers_hf",
        #     index_path="faiss_index.index",
        #     use_dummy_dataset=False,
        #     # dataset_split='train'
        # )
        # retriever.retriever.init_retrieval()

        # rag_model = RagSequenceForGeneration.from_pretrained(
        #     "facebook/rag-sequence-nq",
        #     retriever=retriever
        # ).to(device)


        search_query = input('Enter search query: ')

        # inputs = rag_model.retriever.question_encoder_tokenizer(
        #     search_query, return_tensors="pt"
        # ).to(device)
        # outputs = rag_model.generate(
        #     inputs["input_ids"],
        #     num_beams=5,
        #     num_return_sequences=1
        # )
        # response=search_with_rag(search_query, rag_model, tokenizer)
        # response=rag_model.retriever.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        model_name='google/flan-t5-base'
        generator=T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer=T5Tokenizer.from_pretrained(model_name)

        while True:
            query_type=input(f'\n Do you want to ask question about a specific paper (enter paper no. 1-5) or ask a general question (enter G) or want to exit (enter E) ?')
            if(query_type.lower()=='e'):
                break
            search_query=input('Enter you query :')
            if query_type.lower()=='g':
                print('Using Gemini api for general query...')
                response,sources=gemini_general_query(search_query,main_query)
                print(f'\nResponse from Gemini : {response}')
                # response=general_search_with_rag(search_query,relevent_papers,generator, tokenizer)
                # print(f'Response from RAG : {response}')
                # print(f'\n Sources :')
                # for i,doc in enumerate(sources):
                #     print(f'paper {i+1} :{doc['title']}')
            else:
                try:
                    paper_num=int(query_type)-1
                    response,sources=specific_search_with_rag(paper_num,search_query,relevent_papers,generator, tokenizer)
                    print(f'Response about paper {paper_num+1}: {response}')
                except ValueError:
                    print("Invalid Input ! Pls enter paper no. b/w 1 to 5 or 'G' for general query .")
                    continue
                # continue_search=input('Do you want to ask another question ? (y/n) :')
                # if (continue_search.lower())!='y':
                #     break
    #papers=load_papers()
    # if papers:
    #     summarized_papers=summarize_papers(papers)
    #     print('\n-------SUMMARIZED PAPERS ------')
    #     for paper in sn')'''


def main():

    reset_papers()
    summarizer, generator, tokenizer, embd_model, device = initialize_models()

    gemini_model = setup_gemini()

    main_query = input("Enter your main research topic: ")

    xml_data=fetch_arxiv_papers(main_query)
    if not xml_data:
        print('Failed to fetch xml data. PLs try again..')
        return

    all_papers=parse_xml(xml_data)
    if not all_papers:
        print('No Research papers found for this search. try diff. search term')
        return

    save_papers_to_json(all_papers)

    index,_=create_faiss_index(all_papers,embd_model)

    relevent_papers=search_faiss(main_query,index,all_papers,embd_model)

    display_papers(relevent_papers)

    while True:
        print('\n Do you want to:')
        print('Ask about a spceific paper (Enter paper no. b/w 1-5):')
        print('Ask a general query about main research topic (G):')
        print('Exit(E):')

        choice=input('Enter your choice:').strip()
        search_query = input('Enter your query:')

        if choice.lower()=='e':
            print('OK ,Thank you :)')
            break
        elif choice.lower()=='g':
            print('Using Gemini api for general query...')
            response, sources = gemini_general_query(search_query,main_query,gemini_model,relevent_papers,generator,tokenizer,device)
            print(f'\nResponse from Gemini : {response}')

        else :
            try:
                paper_num = int(choice) - 1
                response, sources = specific_search_with_rag(paper_num,search_query,relevent_papers,generator, tokenizer, device)
                print(f'Response about paper {paper_num + 1}: {response}')
            except ValueError:
                print(f"Invalid Input ! Pls enter paper no. b/w 1 to {len(relevent_papers)} or 'G' or 'E'.")
                continue

if __name__ == "__main__":
    main()






