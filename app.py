import streamlit as st
import json
import os
import base64
import io
from datetime import datetime, timedelta
from typing import List
from PIL import Image, ImageDraw, ImageFont
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import re
import uuid

# Page configuration
st.set_page_config(
    page_title="AI Skincare Assistant",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with updated font colors and box sizes
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 2rem;  /* Increased padding for larger boxes */
        border-radius: 15px;  /* Increased border radius */
        border-left: 4px solid #4ecdc4;
        margin: 1rem 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);  /* Enhanced shadow */
        min-height: 180px;  /* Set minimum height for consistent box sizes */
        color: #2c3e50;  /* Changed font color to dark blue-gray */
    }
    .feature-card h4 {
        color: #34495e;  /* Darker color for headings */
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .feature-card p {
        color: #5d6d7e;  /* Medium gray for paragraph text */
        font-size: 1rem;
        line-height: 1.5;
    }
    .chat-message {
        padding: 1.5rem;  /* Increased padding */
        border-radius: 15px;  /* Increased border radius */
        margin: 0.8rem 10px;  /* Increased margin */
        border-left: 5px solid #4ecdc4;
        min-height: 60px;  /* Minimum height for chat messages */
        font-size: 1rem;
        line-height: 1.4;
    }
    .user-message {
        background-color: #d5e8d4;  /* Lighter green background */
        border-left-color: #27ae60;  /* Changed border color */
        color: #1e3a2e;  /* Dark green text */
    }
    .ai-message {
        background-color: #fadbd8;  /* Lighter red background */
        border-left-color: #e74c3c;  /* Changed border color */
        color: #5d1a1a;  /* Dark red text */
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1.5rem;  /* Increased padding */
        border-radius: 12px;  /* Increased border radius */
        margin: 1rem 0;
        color: #2c3e50;  /* Dark blue-gray text */
        font-size: 0.95rem;
    }
    .sidebar-section strong {
        color: #34495e;  /* Darker color for strong text */
        font-size: 1.1rem;
    }
    
    /* Additional styling for buttons */
    .stButton > button {
        background-color: #4ecdc4;
        color: #ffffff;  /* White text on buttons */
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;  /* Larger button padding */
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        min-height: 50px;  /* Minimum button height */
    }
    .stButton > button:hover {
        background-color: #45b7aa;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
        color: #34495e;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #2c3e50;
        border: 2px solid #e8f4fd;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        color: #2c3e50;
        border: 2px solid #e8f4fd;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
        min-height: 120px;  /* Larger text areas */
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background-color: #f8f9fa;
        border: 2px dashed #4ecdc4;
        border-radius: 12px;
        padding: 2rem;
        color: #2c3e50;
        text-align: center;
        min-height: 150px;
    }
    
    /* Success and error message styling */
    .stSuccess > div {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }
    
    .stError > div {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }
    
    .stWarning > div {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* General text color improvements */
    .stMarkdown {
        color: #2c3e50;
    }
    
    .stMarkdown h3 {
        color: #34495e;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h4 {
        color: #34495e;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_profile = {}
    st.session_state.chat_history = []
    st.session_state.ai_name = "SkinCareBot"
    st.session_state.model = None
    st.session_state.vector_store = None
    st.session_state.message_history = ChatMessageHistory()

@st.cache_resource
def initialize_ai_components():
    """Initialize AI model and vector store (cached for performance)"""
    try:
        # Initialize model
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key="", 
            convert_system_message_to_human=True,
            temperature=0.7
        )
        
        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index_path = "faiss_index"
        
        if os.path.exists(faiss_index_path):
            vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            vector_store = FAISS.from_texts(["Initial empty text"], embeddings)
            vector_store.save_local(faiss_index_path)
        
        return model, vector_store, embeddings
    except Exception as e:
        st.error(f"Error initializing AI components: {str(e)}")
        return None, None, None

# Initialize components
if not st.session_state.initialized:
    model, vector_store, embeddings = initialize_ai_components()
    if model and vector_store:
        st.session_state.model = model
        st.session_state.vector_store = vector_store
        st.session_state.embeddings = embeddings
        st.session_state.initialized = True

# Helper Functions (adapted from your original code)
def add_to_conversation_history(role: str, message: str, user_id: str = None):
    """Add message to conversation history"""
    if not st.session_state.vector_store:
        return
        
    user_id = user_id or st.session_state.user_profile.get("user_id", str(uuid.uuid4()))
    timestamp = datetime.now().isoformat()
    message_id = f"{user_id}_{timestamp}"
    
    try:
        st.session_state.vector_store.add_texts(
            texts=[message],
            metadatas=[{"user_id": user_id, "role": role, "timestamp": timestamp, "message_id": message_id}],
            ids=[message_id]
        )
        st.session_state.vector_store.save_local("faiss_index")
        
        if user_id == st.session_state.user_profile.get("user_id"):
            if role == "user":
                st.session_state.message_history.add_user_message(message)
            else:
                st.session_state.message_history.add_ai_message(message)
            
            if len(st.session_state.message_history.messages) > 10:
                st.session_state.message_history.messages = st.session_state.message_history.messages[-10:]
    except Exception as e:
        st.error(f"Error adding to conversation history: {str(e)}")


def compress_image(image_file, max_size=(800, 800), quality=85) -> bytes:
    """Compress uploaded image"""
    try:
        img = Image.open(image_file)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        return img_byte_arr.getvalue()
    except Exception as e:
        st.error(f"Error compressing image: {str(e)}")
        return None

def create_analysis_prompt(additional_info: str) -> str:
    """Create prompt for skin analysis"""
    user_context = ""
    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        user_context = f"""
User Profile:
- Name: {profile.get('name', 'User')}
- Age: {profile.get('age', 'Unknown')}
- Skin Type: {profile.get('skin_type', 'Unknown')}
- Concerns: {', '.join(profile.get('concerns', []))}
"""
    
    return f"""
You are a dermatology assistant. Analyze the facial image using:
{user_context}
- Additional Info: {additional_info}
- Identify specific skin issues with locations
- Provide short, actionable solutions for each issue
- Format your response clearly with Issues and Solutions sections
- Address user by name if available and provide practical advice
"""

def create_conversation_chain(system_prompt: str):
    """Create conversation chain"""
    if not st.session_state.model:
        return None
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | st.session_state.model
    
    def get_session_history(session_id: str):
        return st.session_state.message_history
    
    return RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

def analyze_skin_image(image_file, additional_info: str = "") -> dict:
    """Analyze uploaded skin image"""
    try:
        compressed_image = compress_image(image_file)
        if not compressed_image:
            return {"success": False, "error": "Failed to process image"}
            
        base64_image = base64.b64encode(compressed_image).decode('utf-8')
        prompt = create_analysis_prompt(additional_info)
        
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ])
        
        response = st.session_state.model.invoke([message])
        
        user_id = st.session_state.user_profile.get("user_id")
        add_to_conversation_history("user", f"Uploaded image for analysis. Info: {additional_info}", user_id)
        add_to_conversation_history("ai", response.content, user_id)
        
        return {
            "success": True,
            "analysis": response.content,
            "image_data": base64_image
        }
    except Exception as e:
        return {"success": False, "error": f"Error analyzing image: {str(e)}"}

def chat_with_ai(user_message: str) -> str:
    """General chat with AI"""
    if not st.session_state.model:
        return "AI model not initialized. Please check your setup."
        
    user_context = ""
    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        user_context = f"""
User Profile:
- Name: {profile.get('name', 'User')}
- Age: {profile.get('age', 'Unknown')}
- Skin Type: {profile.get('skin_type', 'Unknown')}
- Concerns: {', '.join(profile.get('concerns', []))}
"""
    
    system_prompt = f"You are {st.session_state.ai_name}, a supportive skincare assistant. {user_context}"
    
    try:
        chain = create_conversation_chain(system_prompt)
        if not chain:
            return "Error creating conversation chain."
            
        response = chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": st.session_state.user_profile.get("user_id", str(uuid.uuid4()))}}
        ).content
        
        user_id = st.session_state.user_profile.get("user_id")
        add_to_conversation_history("user", user_message, user_id)
        add_to_conversation_history("ai", response, user_id)
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Main App Layout
def main():
    # Header
    st.markdown('<h1 class="main-header">🌟 AI Skincare Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        
        # AI Name Input
        ai_name = st.text_input("Name your AI Assistant:", value=st.session_state.ai_name)
        if ai_name != st.session_state.ai_name:
            st.session_state.ai_name = ai_name
        
        # Profile Creation/Display
        if not st.session_state.user_profile:
            with st.form("profile_form"):
                st.markdown("**Create Your Profile**")
                name = st.text_input("Your Name:")
                age = st.number_input("Age:", min_value=1, max_value=100, value=25)
                skin_type = st.selectbox("Skin Type:", 
                    ["Normal", "Oily", "Dry", "Combination", "Sensitive"])
                concerns = st.multiselect("Skin Concerns:", 
                    ["Acne", "Wrinkles", "Dark Spots", "Dryness", "Oiliness", 
                     "Sensitivity", "Pores", "Redness", "Aging"])
                
                if st.form_submit_button("Create Profile"):
                    if name:
                        user_id = str(uuid.uuid4())
                        st.session_state.user_profile = {
                            "user_id": user_id,
                            "name": name,
                            "age": age,
                            "skin_type": skin_type,
                            "concerns": concerns,
                            "created_at": datetime.now().isoformat()
                        }
                        st.success(f"Profile created for {name}!")
                        st.rerun()
                    else:
                        st.error("Please enter your name.")
        else:
            # Display existing profile
            profile = st.session_state.user_profile
            st.markdown(f"""
            <div class="sidebar-section">
                <strong>👋 {profile['name']}</strong><br>
                Age: {profile['age']}<br>
                Skin Type: {profile['skin_type']}<br>
                Concerns: {', '.join(profile['concerns']) if profile['concerns'] else 'None'}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Reset Profile"):
                st.session_state.user_profile = {}
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Quick Actions")
        if st.button("💡 Get Daily Tip"):
            if st.session_state.user_profile:
                tip = chat_with_ai("Give me a personalized daily skincare tip.")
                st.session_state.chat_history.append(("AI Tip", tip))
            else:
                st.error("Please create a profile first.")
    
    # Main content area
    if not st.session_state.initialized:
        st.warning("⚠️ AI components are initializing. Please wait...")
        return
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📸 Skin Analysis", "🎯 Services", "📋 Recommendations"])
    
    # Chat Tab
    with tab1:
        st.markdown("### Chat with your AI Assistant")
        
        # Chat history display
        if st.session_state.chat_history:
            for role, message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if role.startswith("User"):
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message ai-message"><strong>{st.session_state.ai_name}:</strong> {message}</div>', 
                               unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask me about skincare, routines, products, or anything else!")
        if user_input:
            response = chat_with_ai(user_input)
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append((st.session_state.ai_name, response))
            st.rerun()
    
    # Skin Analysis Tab
    with tab2:
        st.markdown("### 📸 Skin Analysis")
        st.markdown("Upload a clear photo of your face for AI-powered skin analysis.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose an image file", 
                                           type=['png', 'jpg', 'jpeg'], 
                                           help="Upload a clear, well-lit photo of your face")
            
            additional_info = st.text_area("Additional Information (optional):", 
                                         placeholder="Any specific concerns or areas you'd like me to focus on?")
            
            analyze_button = st.button("🔍 Analyze Skin", type="primary")
        
        with col2:
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if analyze_button and uploaded_file is not None:
            with st.spinner("Analyzing your skin..."):
                result = analyze_skin_image(uploaded_file, additional_info)
                
                if result["success"]:
                    st.success("Analysis completed!")
                    st.markdown("### 📊 Analysis Results")
                    st.markdown(result["analysis"])
                    
                    # Add to chat history
                    st.session_state.chat_history.append(("User", f"Uploaded image for analysis. {additional_info}"))
                    st.session_state.chat_history.append((st.session_state.ai_name, result["analysis"]))
                else:
                    st.error(f"Analysis failed: {result['error']}")
    
    # Services Tab
    with tab3:
        st.markdown("### 🎯 Skincare Services")
        
        service_col1, service_col2 = st.columns(2)
        
        with service_col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🧴 Product Recommendations</h4>
                <p>Get personalized product suggestions based on your skin type and concerns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Get Product Recommendations"):
                if st.session_state.user_profile:
                    concerns = ', '.join(st.session_state.user_profile.get('concerns', ['general skincare']))
                    response = chat_with_ai(f"Recommend skincare products for {concerns} with a moderate budget.")
                    st.session_state.chat_history.append(("Product Recommendations", response))
                    st.success("Recommendations added to chat!")
                else:
                    st.error("Please create a profile first.")
            
            st.markdown("""
            <div class="feature-card">
                <h4>📅 Skincare Routine</h4>
                <p>Create a customized daily skincare routine for your needs.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Create Skincare Routine"):
                if st.session_state.user_profile:
                    skin_type = st.session_state.user_profile.get('skin_type', 'normal')
                    concerns = ', '.join(st.session_state.user_profile.get('concerns', ['general care']))
                    response = chat_with_ai(f"Create a beginner-level skincare routine for {skin_type} skin addressing {concerns}.")
                    st.session_state.chat_history.append(("Skincare Routine", response))
                    st.success("Routine added to chat!")
                else:
                    st.error("Please create a profile first.")
        
        with service_col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🔬 Ingredient Analysis</h4>
                <p>Analyze product ingredients for compatibility and effectiveness.</p>
            </div>
            """, unsafe_allow_html=True)
            
            ingredients_input = st.text_area("Enter product ingredients:", 
                                           placeholder="List the ingredients you want to analyze...")
            if st.button("Analyze Ingredients") and ingredients_input:
                response = chat_with_ai(f"Please analyze these skincare ingredients: {ingredients_input}")
                st.session_state.chat_history.append(("Ingredient Analysis", response))
                st.success("Analysis added to chat!")
            
            st.markdown("""
            <div class="feature-card">
                <h4>🍎 Lifestyle & Diet</h4>
                <p>Get advice on how lifestyle and diet affect your skin health.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Get Lifestyle Advice"):
                if st.session_state.user_profile:
                    age = st.session_state.user_profile.get('age', 25)
                    concerns = ', '.join(st.session_state.user_profile.get('concerns', ['general health']))
                    response = chat_with_ai(f"Provide lifestyle and dietary recommendations for healthy skin. I'm {age} years old with concerns about {concerns}.")
                    st.session_state.chat_history.append(("Lifestyle Advice", response))
                    st.success("Advice added to chat!")
                else:
                    st.error("Please create a profile first.")
    
    # Recommendations Tab
    with tab4:
        st.markdown("### 📋 Personalized Recommendations")
        
        if st.session_state.user_profile:
            profile = st.session_state.user_profile
            
            st.markdown(f"#### Recommendations for {profile['name']}")
            st.markdown(f"**Skin Type:** {profile['skin_type']} | **Age:** {profile['age']}")
            
            # Quick recommendation buttons
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                if st.button("🌅 Morning Routine"):
                    response = chat_with_ai("What should my morning skincare routine be?")
                    st.markdown("**Morning Routine Recommendation:**")
                    st.markdown(response)
            
            with rec_col2:
                if st.button("🌙 Night Routine"):
                    response = chat_with_ai("What should my evening/night skincare routine be?")
                    st.markdown("**Night Routine Recommendation:**")
                    st.markdown(response)
            
            with rec_col3:
                if st.button("🍽️ Diet Tips"):
                    response = chat_with_ai("What foods should I eat for better skin health?")
                    st.markdown("**Diet Recommendations:**")
                    st.markdown(response)
            
            # Custom recommendation request
            st.markdown("---")
            custom_request = st.text_input("Ask for specific recommendations:", 
                                         placeholder="e.g., products for sensitive skin, anti-aging routine...")
            if st.button("Get Custom Recommendation") and custom_request:
                response = chat_with_ai(custom_request)
                st.markdown("**Custom Recommendation:**")
                st.markdown(response)
        else:
            st.warning("Please create your profile first to get personalized recommendations.")

if __name__ == "__main__":

    main()

