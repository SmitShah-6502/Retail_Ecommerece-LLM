🛍️ AI-Based Product Recommender for Retail & E-Commerce
A cutting-edge web application built with Streamlit and Python, leveraging Google Gemini and advanced AI techniques to provide personalized product recommendations, multilingual chatbot assistance, and inventory insights for an enhanced e-commerce experience.

Features
Product Search: Enter text queries (e.g., "red cotton shirt") to receive 5–10 AI-generated product listings powered by Google Gemini 1.5 Flash, including name, price, description, sizes, colors, gender, and URL.
Image-Based Product Finder: Upload an image to get a tailored product recommendation based on Gemini’s multimodal analysis, with optional text hints for precision.
Multilingual Chatbot Assistance: Supports English, Hindi, and Gujarati queries via text or voice, using DistilGPT2 for responses, Google Translator for NLP, and gTTS for text-to-speech.
Product Comparison: Compares products by price, rating, and features, with AI-driven category detection and mock data generation for realistic insights.
Inventory Management: Displays top-selling products, most-viewed items, low-stock alerts, and category rankings using pandas, with Excel export functionality.
Modern UI: Responsive Streamlit interface with custom CSS, Lottie animations, and a vibrant color scheme for an engaging user experience.

Tech Stack
Frontend: Streamlit, HTML, CSS (custom styling), streamlit-lottie (animations)
Backend: Python, pandas, Flask (optional for deployment)
AI Integration:
Google Gemini 1.5 Flash (LLM for product search and image analysis)
DistilGPT2 (chatbot responses)
Google Translator (NLP for multilingual support)
gTTS and speech_recognition (text-to-speech and voice input)


Libraries: PIL, numpy, xlsxwriter, hashlib, requests
Datasets: Sample product data (mock inventory with sales, views, and stock metrics)

Project Structure
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

Installation & Run Locally
1. Clone the Repository
git clone https://github.com/your-username/ai-product-recommender.git
cd ai-product-recommender

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Add Your Gemini API Key
Open app.py and replace the placeholder with your Gemini API key:
genai.configure(api_key="YOUR_API_KEY")

5. Run the Streamlit App
streamlit run app.py

Visit http://127.0.0.1:8501 in your browser.

How It Works
Product Search: Uses Gemini 1.5 Flash to generate product listings based on text queries, parsed and displayed in styled cards with custom CSS and emojis.
Image-Based Product Finder: Analyzes uploaded images with Gemini’s multimodal capabilities, generating a single recommendation with detailed attributes.
Multilingual Chatbot: Processes text/voice queries in English, Hindi, or Gujarati, leveraging DistilGPT2 and Google Translator for seamless interactions.
Product Comparison: Generates mock product data with category-specific features, displaying comparisons in a visually appealing format.
Inventory Management: Analyzes sample data with pandas to show top-selling products, low-stock alerts, and category rankings, with an Excel download option.
UI Enhancements: Features Lottie animations, a blue-gray color scheme, and responsive layouts for a modern e-commerce experience.

Future Enhancements
Integrate real-time product APIs for live catalog data.
Deploy on cloud platforms like Render or Heroku.
Add user-based personalization using collaborative filtering.
Enhance voice interaction with advanced speech recognition.
Expand inventory analytics with predictive stock forecasting.

Credits
Google Gemini 1.5 Flash: For product search and image analysis.
Hugging Face (DistilGPT2): For chatbot responses.
Google Translator & gTTS: For multilingual support and text-to-speech.
Streamlit & streamlit-lottie: For the interactive UI and animations.

📄 License
This project is open-source under the MIT License.
