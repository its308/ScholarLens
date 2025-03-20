import streamlit as st
import main
import torch
import os
import json

PORT = int(os.environ.get("PORT", 10000))


st.run(host="0.0.0.0", port=PORT)
st.set_page_config(page_title="ScholarLens", page_icon="📚", layout="wide")


st.title("ScholarLens : AI Research Assistant")
st.markdown("Search academic papers and get AI-powered insights about research topics")



@st.cache_resource
def load_models():
    summarizer, generator, tokenizer, embd_model, device = main.initialize_models()
    gemini_model = main.setup_gemini(api_key="AIzaSyCFjHwUdOGdFSCH7fKXFz4Ljr3i9df0piA")
    return summarizer, generator, tokenizer, embd_model, device, gemini_model



if 'models_loaded' not in st.session_state:
    st.session_state.summarizer, st.session_state.generator, st.session_state.tokenizer, \
        st.session_state.embd_model, st.session_state.device, st.session_state.gemini_model = load_models()
    st.session_state.models_loaded = True



def reset_session():
    if 'papers' in st.session_state:
        del st.session_state.papers
    if 'relevant_papers' in st.session_state:
        del st.session_state.relevant_papers
    if 'search_performed' in st.session_state:
        st.session_state.search_performed = False


    try:
        with open('papers.json', 'w') as file:
            json.dump([], file)
        print('Reset papers.json to an empty list')
    except Exception as e:
        print(f"Error resetting papers file: {e}")



with st.sidebar:
    st.header("About")
    st.write(
        "This AI Research Assistant helps you explore academic papers and get AI-powered answers to your research questions.")

    st.header("Options")
    if st.button("Clear Results"):
        reset_session()
        st.rerun()

    st.header("Credits")
    st.write("Built with Streamlit, Hugging Face, and Google Gemini")
    st.write("© 2024")


search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    topic = st.text_input("Enter a research topic:", value="", key="search_box")
with search_col2:
    search_button = st.button("Search Papers", use_container_width=True)


if search_button and topic:

    reset_session()


    status_container = st.status("Starting search...")


    status_container.update(label="Fetching papers from ArXiv...", state="running")
    xml_data = main.fetch_arxiv_papers(topic)

    if xml_data:

        status_container.update(label="Processing paper data...", state="running")
        all_papers = main.parse_xml(xml_data)

        if all_papers:

            try:
                with open('papers.json', 'w') as file:
                    json.dump(all_papers, file, indent=4)
            except Exception as e:
                st.error(f"Error saving papers: {e}")


            status_container.update(label="Creating paper embeddings...", state="running")
            index, _ = main.create_faiss_index(all_papers, st.session_state.embd_model)
            relevant_papers = main.search_faiss(topic, index, all_papers, st.session_state.embd_model)


            st.session_state.papers = all_papers
            st.session_state.relevant_papers = relevant_papers
            st.session_state.search_performed = True
            st.session_state.topic = topic

            status_container.update(label="Papers found and processed successfully!", state="complete")
        else:
            status_container.update(label="No papers found. Try a different search term.", state="error")
            st.error("No papers found. Try a different search term.")
    else:
        status_container.update(label="Failed to fetch papers. Please try again later.", state="error")
        st.error("Failed to fetch papers. Please try again later.")


if 'search_performed' in st.session_state and st.session_state.search_performed:
    st.header(f"Top 5 Papers on '{st.session_state.topic}'")


    tabs = st.tabs([f"Paper {i + 1}" for i in range(len(st.session_state.relevant_papers))])


    for i, (tab, paper) in enumerate(zip(tabs, st.session_state.relevant_papers)):
        with tab:
            st.subheader(paper['title'])
            st.write(f"**Authors:** {', '.join(paper['authors'])}")


            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Abstract:**")
                st.write(paper['summary'])
            with col2:
                st.markdown("**Concise Summary:**")
                summary = main.generate_summary(paper['summary'], st.session_state.summarizer)
                st.write(summary)


            keywords = main.extract_keywords(paper['summary'])
            st.write(f"**Keywords:** {', '.join(keywords)}")


    st.header("Ask Questions")

    query_col1, query_col2 = st.columns([1, 3])

    with query_col1:
        query_type = st.selectbox(
            "Question type:",
            options=["General Query"] + [f"About Paper {i + 1}" for i in range(len(st.session_state.relevant_papers))],
            index=0
        )

    with query_col2:
        query = st.text_input("Enter your question:", key="query_input")

    if st.button("Get Answer", use_container_width=True):
        if query:

            if query_type == "General Query":
                st.subheader("Response")


                gen_status = st.status("Generating response ...", state="running")


                response, sources = main.gemini_general_query(
                    query,
                    st.session_state.topic,
                    st.session_state.gemini_model,
                    papers=st.session_state.relevant_papers,
                    generator=st.session_state.generator,
                    tokenizer=st.session_state.tokenizer,
                    device=st.session_state.device
                )


                if sources:
                    gen_status.update(label="Response generated using local RAG (Gemini API unavailable)",
                                      state="complete")
                else:
                    gen_status.update(label="Response generated ", state="complete")


                st.markdown(response)


            else:
                paper_num = int(query_type.split(" ")[-1]) - 1
                st.subheader(f"Response about Paper {paper_num + 1}")


                paper_status = st.status("Generating paper-specific response...", state="running")

                response, _ = main.specific_search_with_rag(
                    paper_num,
                    query,
                    st.session_state.relevant_papers,
                    st.session_state.generator,
                    st.session_state.tokenizer,
                    st.session_state.device
                )

                paper_status.update(label="Response generated", state="complete")
                st.write(response)
        else:
            st.warning("Please enter a question.")
else:

    st.info("Enter a research topic above to get started.")
    st.markdown("""
    ## How to use this Research Assistant

    1. Enter a scientific research topic in the search box above
    2. Browse the top 5 papers retrieved from ArXiv
    3. Ask general questions about the topic or specific questions about individual papers
    4. Get AI-powered answers based on the papers or broader knowledge

    This tool helps you explore research topics and understand academic papers more efficiently.
    """)


    st.markdown("### Try these sample topics:")
    sample_topics = [
        "Quantum Computing Algorithms",
        "Machine Learning Healthcare",
        "Climate Change Models",
        "Graphene Applications Electronics",
        "Protein Folding Prediction"
    ]


    cols = st.columns(len(sample_topics))
    for i, topic in enumerate(sample_topics):
        with cols[i]:
            if st.button(topic, key=f"sample_{topic}"):
                st.session_state.search_box = topic
                st.rerun()