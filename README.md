# ET NEWS INTELLIGENCE
## Project Documentation
### Current Phase 1 Implementation

 1. Project Overview
ET News Intelligence is a news research application built around Economic Times data. The current Phase 1 system lets users ask natural-language questions about the available ET articles and receive evidence-based answers. It also lets users explore articles by topic and date range.
The system first searches the ET dataset for relevant articles and then gives the retrieved evidence to a language model. This is the project's Retrieval-Augmented Generation (RAG) approach.
2. What We Built
Component	Status	Details
ET dataset	Completed	22,354 Economic Times articles
Data cleaning	Completed	Missing values, duplicates, text and dates handled
Embeddings	Completed	22,354 vectors, 1,024 dimensions
Semantic search	Completed	Top 5 relevant articles retrieved
RAG	Completed	Retrieved ET evidence supplied to Mistral
Evidence rules	Completed	Outside knowledge and invented facts are discouraged
Unsupported questions	Completed	System can say evidence is insufficient
Supporting sources	Completed	Article titles, metadata and links displayed
Streamlit UI	Completed	Simple white, black and bold-blue interface
Explore News	Completed	Topic and date-range filtering
Basic testing	Completed	Normal, complex, multi-article and unsupported questions tested
3. Dataset
The current application uses:
Data/articles_clean.csv
The dataset contains 22,354 articles and these fields:
•	Topic
•	Title
•	Article_URL
•	Publication_Date
•	Extraction_Date
•	Synopsis
4. Data Processing
The data preparation work included checking missing values, duplicate rows and duplicate article URLs, cleaning text fields, and converting publication and extraction dates. The cleaned dataset is used by the current search and RAG system.
5. Article Embeddings
Each article was converted into a numerical embedding using the Ollama model:
qwen3-embedding:0.6b
The resulting embedding matrix contains 22,354 vectors with 1,024 dimensions.
models/article_embeddings.npy
Checkpoint and index files were also created during the embedding process.
6. Semantic Search
Main file:
src/semantic_search.py
Search process:
1.	User enters a question.
2.	The question is converted into an embedding.
3.	The question vector is compared with the saved article embeddings.
4.	Similarity scores are calculated.
5.	The top 5 articles are returned.
7. RAG System
Main file:
src/rag_system.py
The current RAG process:
6.	Receive the user's question.
7.	Retrieve relevant Economic Times articles.
8.	Build an evidence context from title, topic, date and synopsis.
9.	Send the question and retrieved evidence to mistral:latest through Ollama.
10.	Display the generated answer and supporting ET sources.
8. Evidence-Based Answering
•	Use only the supplied Economic Times evidence.
•	Do not use outside knowledge.
•	Do not invent facts.
•	Say when the evidence is insufficient.
•	Keep answers clear and concise.
•	Do not place URLs inside the generated answer.
The unsupported-question test using an Apple factory on Mars showed the intended refusal behavior.
9. Streamlit Application
app/app.py
9.1 Ask Question
Users enter a natural-language question. The application performs semantic retrieval, generates a RAG answer, and displays supporting Economic Times sources.
9.2 Explore News by Time
Users can select a topic and a date range. Matching articles are displayed from newest to oldest, with links to the original articles.
9.3 Interface Design
•	White background
•	Black text
•	Bold blue accent
•	Large bold headings
•	Simple editorial-style layout
•	No sidebar
•	No gradient-heavy dashboard
•	No chatbot bubble
10. Technologies Used
Technology	Purpose
Python	Main programming language
Pandas	Data loading and processing
NumPy	Embeddings and similarity calculations
Ollama	Local model execution
qwen3-embedding:0.6b	Article and question embeddings
mistral:latest	RAG answer generation
Streamlit 1.61.1	Web application
mysql-connector-python	Python MySQL connector installed during setup
11. Main Project Structure
ET News/
├── Data/
│   └── articles_clean.csv
├── models/
│   ├── article_embeddings.npy
│   ├── embedding_checkpoint.npy
│   └── embedding_indices.npy
├── src/
│   ├── semantic_search.py
│   ├── rag_system.py
│   ├── create_embeddings.py
│   ├── clean_data.py
│   ├── combine_data.py
│   ├── check_data.py
│   ├── inspect_articles.py
│   └── test_classification.py
├── app/
│   └── app.py
└── .venv/
12. How to Run the Current Project
Open PowerShell and run:
cd "C:\Users\HP\Desktop\ET News"
.\.venv\Scripts\Activate.ps1
Check Ollama models if needed:
ollama list
Start the application:
streamlit run app\app.py
Open the local Streamlit address shown in the terminal, normally http://localhost:8501.
13. What to Expect
•	The application runs locally in a browser.
•	Ask Question searches the saved 22,354-article ET dataset.
•	The question is embedded before retrieval.
•	The top 5 matching articles are used as RAG evidence.
•	Mistral generates the answer from that evidence.
•	Supporting ET article links are displayed.
•	Explore News filters the same dataset by topic and date.
•	AI responses can take several seconds because the models run locally.
•	New ET articles do not automatically appear in the current AI search.
14. Testing Performed
•	AI and business questions
•	AI and workplace questions
•	AI adoption challenges
•	Effects of AI on Indian companies
•	Multi-article questions about strategy and AI opportunities/risks
•	Cybersecurity question
•	Unsupported question about an Apple factory on Mars
The tests showed useful retrieval, multi-article evidence use, source display, and correct refusal when sufficient evidence was unavailable.
15. RSS and MySQL Work Already Created
Separate files were provided for Economic Times RSS collection and MySQL storage, including fetch_all_RSS.py, test.RSS.py, Database_Connection.py, and several category-specific import scripts.
These scripts contain work for fetching ET RSS feeds, processing article data, and inserting records into a MySQL news table.
This RSS/MySQL work is currently separate from the Streamlit/RAG pipeline.
16. Current MySQL Status
mysql-connector-python was installed successfully. However, the local MySQL server could not be reached on localhost:3306, and no MySQL service or mysql command was found during the checks. Therefore MySQL is not currently connected to the running Streamlit/RAG application.
17. Not Yet Implemented
•	Live RSS data connected to the current RAG system
•	MySQL connected to the current RAG system
•	Automatic embedding of newly collected articles
•	Automatic daily updates
•	Online deployment
•	Phase 2 event detection or news timelines
18. Current Limitations
•	The current AI search works from the saved 22,354-article dataset.
•	New ET articles do not automatically appear in AI search.
•	Answer quality depends on the relevance and quality of retrieved articles.
•	Semantic search can sometimes retrieve articles that are related but do not fully answer the question.
•	Local model speed depends on the computer.
•	A full quantitative evaluation has not yet been completed.
19. Conclusion
The current project is a working Phase 1 prototype of an Economic Times News Intelligence System. It combines a cleaned dataset of 22,354 ET articles, 1,024-dimensional embeddings, semantic search, RAG, Mistral answer generation, supporting source display, and a Streamlit interface.
The main achievement is that users can ask natural-language questions and receive answers based on retrieved Economic Times evidence instead of relying only on general model knowledge. Users can also explore articles by topic and date.
The project also contains separate RSS and MySQL work that can support a future live-data pipeline, but that pipeline is not connected to the current AI application.
20. Quick Reference
Item	Current Value
Project	ET News Intelligence
Articles	22,354
Embedding model	qwen3-embedding:0.6b
Embedding size	1,024 dimensions
LLM	mistral:latest
Retrieval	Semantic search, top 5
RAG	Yes
Web UI	Streamlit 1.61.1
Main app	app/app.py
Dataset	Data/articles_clean.csv
Embeddings	models/article_embeddings.npy
Current data mode	Saved dataset
Live RSS/MySQL	Separate, not connected to RAG
