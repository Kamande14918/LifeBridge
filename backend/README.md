This chatbot is majorly powered by hugging face pipelines , that is the sentence-transformers/all-MiniLM-L6-v2 and the llm used is the google/flan-t5-base. The responses of the chatbot are grounded using the sample first aid cases in the which are vectorized and indexed using the faiss vector database.

<!-- Installation instructions -->
1. Clone the repo: git clone: git  https://github.com/Kamande14918/LifeBridge.
2. Install all the dependencies using: pip install -r requirements.txt
3. Build the faiss index using: python build_index.py
4. run server using: python server.py