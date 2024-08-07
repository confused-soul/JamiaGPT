# JamiaGPT 

JamiaGPT is a web-based chat application powered by AI, designed to provide information and engage in roleplay conversations related to Jamia Millia Islamia.
[🔗JamiaGPT](https://jamiagpt.streamlit.app/)

## Video Demonstration

<div>
  <a href="https://drive.google.com/file/d/1m80Bj13fjiuJ9SfgRy6I7E6wrmIyG6UG/view?usp=sharing" target="_blank">
   <p> Watch the Video!</p>
  </a>
</div>


## Screenshots

<p align="center">
  <img src="src/images/JamiaGPT(1).png" width="550" title="Screenshot(1)" alt="[screenshot1](src/images/JamiaGPT(1).png?raw=true)">
  <img src="src/images/JamiaGPT(2).png" width="550" title="Screenshot(2)" alt="[screenshot2](src/images/JamiaGPT(2).png?raw=true)">
</p>

## Features

- **AI Chat Interface:** Engage in roleplay conversations with JamiaGPT.
- **Information Retrieval:** Retrieve relevant information based on user queries.
- **Disclaimer:** Warns users about the model's current development stage and potential limitations.

## Technologies Used

- **Streamlit:** Web framework for creating interactive web applications with Python.
- **LangChain:** Library for building AI chat applications, incorporating:
  - **GoogleGenerativeAI:** For generating AI responses. LLM used : gemini-1.5-flash
  - **HuggingFace Transformers:** Used for embeddings and inference. Embedding Model used : all-MiniLM-l6-v2
  - **FAISS:** (Facebook AI Similarity Search) is a library for fast similarity search from the Vector Database and handling document retrieval.
- **Python Libraries:** Includes some Langchain libraries for data storage, management, and manipulation.

<p align="center">
  <img src="src/images/RAG-Model.jpg" width="550" title="RAG Model" alt="[RAG Model](src/images/RAG-Model.jpg?raw=true)">
</p>

## Setup Instructions

1. **Clone the Repository:**
```bash
git clone https://github.com/confused-soul/JamiaGPT.git
cd JamiaGPT
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set Up Secrets:**
- Ensure you have configured your secrets for Gemini API and Inference API keys in your Streamlit secrets manager.

4. **Run the Application:**
```bash
streamlit run app.py
```

5. **Interact with JamiaGPT:**
- Open your web browser and navigate to the provided local host URL.
- Start interacting by entering queries related to Jamia Millia Islamia.

## Usage

- **Chat Interface:** Enter questions or engage in roleplay with JamiaGPT.
- **Feedback:** Provide feedback on responses to improve the model.

## Contributing

Contributions are welcome! Please follow the standard guidelines for contributions and open issues for feature requests or bug reports.
