# indian-food_nutrection-rag-chatbot
Bilingual AI-powered Indian Food and Nutrition Assistant using RAG
## Project Architecture

![Indian Food Assistant RAG Architecture](indian_food_assistant_rag_architecture.png)
# 🍲 Indian Food & Nutrition Assistant – RAG Chatbot

A bilingual AI-powered Indian Food and Nutrition Assistant built using Retrieval-Augmented Generation (RAG). It helps users explore food information, nutritional values, and cooking recipes in both Tamil and English.

## 🏗️ Project Architecture

![Indian Food Assistant RAG Architecture](indian_food_assistant_rag_architecture.png)

## ✨ Key Features

- **Bilingual Support:** Supports Tamil and English questions.
- **RAG Architecture:** Retrieves relevant food information from a vector database before generating answers.
- **Semantic Search:** Uses multilingual embeddings to understand user queries.
- **Nutrition Information:** Provides food nutritional details based on the available dataset.
- **Recipe Assistance:** Explains cooking recipes and ingredients.
- **Conversation Memory:** Supports follow-up questions using previous conversation context.
- **Source Retrieval:** Displays retrieved food sources.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Chatbot interface |
| Retrieval-Augmented Generation (RAG) | Knowledge-grounded responses |
| Multilingual-E5-Base | Text embeddings |
| ChromaDB | Vector database |
| Groq API | LLM response generation |
| GPT-OSS-20B | Language model |
| JSON | Food dataset |

## 🔄 Working Flow

1. User enters a question in Tamil or English.
2. The question is converted into an embedding using Multilingual-E5-Base.
3. ChromaDB searches for the top matching food information chunks.
4. Relevant food records and their chunks are retrieved.
5. Retrieved context and recent conversation history are passed to the Groq LLM.
6. The LLM generates a response in the appropriate language.
7. Streamlit displays the final answer and retrieved food sources.

## 📂 Dataset Categories

- Fruits
- Vegetables
- Meat
- Seafood
- Food Recipes

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/poovendiran123/indian-food_nutrition-rag-chatbot.git
```

### 2. Navigate to Project Directory

```bash
cd indian-food_nutrition-rag-chatbot
```

### 3. Install Dependencies

```bash
pip install streamlit chromadb groq python-dotenv sentence-transformers numpy
```

### 4. Configure API Key

Create a local file named `ask_key.env` in the project directory.

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload this file to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

## 📌 Database Preparation

The application uses a persistent ChromaDB database.

- Database directory: `chroma_food_db`
- Collection name: `indian_food_bilingual_v1`
- Embedding model: `intfloat/multilingual-e5-base`

Run the data preparation and vector storage workflow in `food.ipynb` to create the required database before starting the application.

**Note:** The notebook currently references additional input and generated files such as `food_recipes.json`, `food_chunks.json`, and `food_embeddings (2).npy`. These need to be available or regenerated locally before executing the corresponding notebook cells.

## 👨‍💻 Project Information

**Project:** Indian Food & Nutrition Assistant

**Category:** Generative AI | RAG | NLP

**Interface:** Streamlit

**Languages:** Tamil & English

---
