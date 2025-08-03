# YourSelfBeauty: AI Skincare Assistant

This project is an interactive AI-powered skincare assistant implemented in a Jupyter Notebook. It leverages advanced language models, vector search, and image processing to provide personalized skincare advice, product recommendations, and lifestyle suggestions.

## Features
- **Conversational AI:** Chat with a supportive skincare assistant for advice and information.
- **Personalized Recommendations:** Get routines, products, and tips tailored to your skin type and concerns.
- **Image Analysis:** Upload facial images for AI-powered skin analysis and annotated feedback.
- **History & Memory:** Maintains conversation history and summarizes long-term interactions for context-aware responses.
- **Lifestyle & Meal Planning:** Receive dietary and lifestyle suggestions for holistic skin health.
- **Meeting Scheduling:** Schedule reminders or meetings for skincare routines or consultations.

## Technologies Used
- **Python** (Jupyter Notebook)
- **LangChain** (for LLM orchestration)
- **Google Gemini** (LLM API)
- **FAISS** (vector store for memory)
- **HuggingFace Embeddings** (semantic search)
- **Pillow** (image processing)

## How to Use
1. **Install Requirements:**
   - Install dependencies from `requirements.txt` using `pip install -r requirements.txt`.
2. **Run the Notebook:**
   - Open `AIChat.ipynb` in Jupyter or VS Code.
   - Execute cells sequentially.
3. **Interact:**
   - Name your AI assistant when prompted.
   - Create a user profile for personalized results.
   - Use natural language to ask for routines, product advice, upload images, or request tips.
   - Type `bye` to end the session.

## Example Queries
- "Create a skincare routine for oily skin."
- "Recommend products for acne-prone skin."
- "Analyze my face image."
- "Give me a daily skincare tip."
- "Schedule a meeting with my dermatologist."

## Notes
- Requires a valid Google Gemini API key for LLM features.
- All user data and conversation history are stored locally in the FAISS vector store.
- For best results, use clear and specific queries.

## License
This project is for educational and personal use. Please review third-party library licenses for commercial applications.
