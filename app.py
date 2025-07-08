import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
from transformers import pipeline
from gtts import gTTS
import speech_recognition as sr
import random
import hashlib

# --- Product Search Feature ---
def product_search():
    # Configure Gemini API key
    genai.configure(api_key="AIzaSyC8Qamg9uhd-jfUtMQ1fbyZ3CXwarfziKA")  # 🔑 Replace with your actual key

    # Load Gemini 1.5 Flash model
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    st.title("🛍️Product Finder")

    # Input query
    query = st.text_input("Enter your product search:", "Red cotton shirt size M")

    # On button click
    if st.button("Search Product"):
        st.markdown("## 🛒 AI-Recommended Products")

        prompt = f"""
        You are an e-commerce assistant. Based on the product query "{query}", generate 5 to 10 product listings.
        Each product should include:
        - Product Name
        - Price (₹)
        - Product Description (1-2 lines)
        - Sizes Available
        - Colors Available
        - Gender (Men, Women, Unisex)
        - Product URL

        Format:
        Product 1:
        Name: ...
        Price: ...
        Description: ...
        Sizes Available: ...
        Colors Available: ...
        Gender: ...
        Link: ...

        Give only product listings, no additional text or headings.
        """

        # Generate content
        response = model.generate_content(prompt)
        text = response.text

        # Parse and display products
        products = []
        blocks = text.split("Product ")
        for block in blocks[1:]:
            lines = block.strip().split("\n")
            product = {}
            for line in lines:
                if "Name:" in line:
                    product["name"] = line.split("Name:")[1].strip()
                elif "Price:" in line:
                    product["price"] = line.split("Price:")[1].strip()
                elif "Description:" in line:
                    product["description"] = line.split("Description:")[1].strip()
                elif "Sizes Available:" in line:
                    product["sizes"] = line.split("Sizes Available:")[1].strip()
                elif "Colors Available:" in line:
                    product["colors"] = line.split("Colors Available:")[1].strip()
                elif "Gender:" in line:
                    product["gender"] = line.split("Gender:")[1].strip()
                elif "Link:" in line:
                    product["link"] = line.split("Link:")[1].strip()
            if all(k in product for k in ["name", "price", "description", "sizes", "colors", "gender", "link"]):
                products.append(product)

        # Display each product
        for p in products:
            st.subheader(p["name"])
            st.write(f"💰 **Price**: {p['price']}")
            st.write(f"📝 **Description**: {p['description']}")
            st.write(f"📏 **Sizes Available**: {p['sizes']}")
            st.write(f"🎨 **Colors Available**: {p['colors']}")
            st.write(f"🚻 **Gender**: {p['gender']}")
            st.markdown(f"[🛒 Buy Now]({p['link']})", unsafe_allow_html=True)

# --- Image-Based Product Finder Feature ---
def image_description():
    # Configure Gemini API Key
    genai.configure(api_key="AIzaSyC8Qamg9uhd-jfUtMQ1fbyZ3CXwarfziKA")  # Replace with your actual key

    # Load the Gemini 1.5 Flash model
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    st.title("🖼️ Image-Based Product Finder")

    st.markdown("""
    Upload a product image and optionally add a caption. The system will analyze the image with Gemini and return a single recommended product based on your image and hint.
    """)

    # File uploader (no pre-stored catalog images needed)
    uploaded_image = st.file_uploader("Upload an image of a product (e.g., shirt, shoes)", type=["jpg", "png", "jpeg"])
    user_hint = st.text_input("Optional: Add a hint or short description (e.g., 'red cotton shirt for men')")

    # Helper function: Use Gemini to generate description from the image (with optional hint)
    def describe_image(image_bytes, hint=None):
        image = Image.open(io.BytesIO(image_bytes))
        parts = [image]
        if hint:
            parts.append(f"This image seems to be a {hint}. Please describe the product in detail.")
        else:
            parts.append("Describe the product in detail.")
        response = model.generate_content(parts)
        return response.text.strip()

    # Helper function: Get product recommendation based on description
    def get_product_recommendation(description):
        prompt = f"""
    You are an expert e-commerce assistant.

    Based on the following product description:
    "{description}"

    Recommend one similar product with the following details:
    - Name
    - Price (₹)
    - Product Description (1-2 lines)
    - Sizes Available
    - Colors Available
    - Gender (Men, Women, Unisex)
    - Product URL

    Format:
    Product 1:
    Name: ...
    Price: ...
    Description: ...
    Sizes Available: ...
    Colors Available: ...
    Gender: ...
    Link: ...

    Respond only with the product details.
    """
        response = model.generate_content(prompt)
        return response.text

    # Main flow
    if uploaded_image and st.button("🔍 Product Description"):
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        img_bytes = uploaded_image.read()

        with st.spinner("Describing the product..."):
            product_description = describe_image(img_bytes, user_hint)
        st.success("✅ Product described.")
        st.markdown(f"### AI Description:\n{product_description}")

        with st.spinner("Fetching product recommendation..."):
            rec_text = get_product_recommendation(product_description)
        
        # Parse the recommendation text to extract one product's details
        product = {}
        # Expecting format with "Product 1:" at the start
        blocks = rec_text.split("Product ")
        if len(blocks) > 1:
            # Process only the first product listing from the output
            block = blocks[1]
            lines = block.strip().split("\n")
            for line in lines:
                if "Name:" in line:
                    product["name"] = line.split("Name:")[1].strip()
                elif "Price:" in line:
                    product["price"] = line.split("Price:")[1].strip()
                elif "Description:" in line:
                    product["description"] = line.split("Description:")[1].strip()
                elif "Sizes Available:" in line:
                    product["sizes"] = line.split("Sizes Available:")[1].strip()
                elif "Colors Available:" in line:
                    product["colors"] = line.split("Colors Available:")[1].strip()
                elif "Gender:" in line:
                    product["gender"] = line.split("Gender:")[1].strip()
                elif "Link:" in line:
                    product["link"] = line.split("Link:")[1].strip()

        if product and all(k in product for k in ["name", "price", "description", "sizes", "colors", "gender", "link"]):
            st.markdown("## 🛍️ Recommended Product")
            st.subheader(product["name"])
            st.write(f"💰 **Price**: {product['price']}")
            st.write(f"📝 **Description**: {product['description']}")
            st.write(f"📏 **Sizes Available**: {product['sizes']}")
            st.write(f"🎨 **Colors Available**: {product['colors']}")
            st.write(f"🚻 **Gender**: {product['gender']}")
            st.markdown(f"[🛒 Buy Now]({['link']})", unsafe_allow_html=True)
        else:
            st.error("❌ Unable to parse product recommendation details. Please try again.")
    else:
        st.info("📷 Upload an image to start.")

# --- Multilingual Chatbot Assistance Feature ---
def multilingual_chatbot_assistance():
    # Load model
    chatbot = pipeline("text-generation", model="distilgpt2")

    # Language codes
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Gujarati": "gu"
    }

    # Translate to English
    def translate_to_english(text, source_lang_code):
        if source_lang_code != "en":
            translated = GoogleTranslator(source=source_lang_code, target="en").translate(text)
            return translated
        return text

    # Translate from English to user language
    def translate_back(text, target_lang_code):
        if target_lang_code != "en":
            translated = GoogleTranslator(source="en", target=target_lang_code).translate(text)
            return translated
        return text

    # Generate detailed response
    def handle_query_detailed_en(query):
        query = query.lower()
        if "order" in query or "parcel" in query:
            return (
                "Thank you for your query regarding your order. Your package is currently being processed and will "
                "be dispatched soon. You can track the order status from your order history. Estimated delivery is within 3–5 business days."
            )
        elif "return" in query:
            return (
                "Returns are hassle-free! You can return your order within 7 days of delivery by going to the Returns section. "
                "Make sure the product is unused and in its original packaging."
            )
        elif "availability" in query or "stock" in query:
            return (
                "The product you're looking for is currently available. However, due to high demand, availability may change. "
                "Please place your order soon to ensure you receive the item on time."
            )
        elif "delivery" in query:
            return (
                "We offer standard delivery within 3–5 business days. You will receive a tracking link once your order is shipped. "
                "Thank you for shopping with us!"
            )
        else:
            response = chatbot(query, max_length=100, do_sample=True)[0]['generated_text']
            return response.strip()

    # Voice input using microphone
    def recognize_speech():
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                with st.spinner("Listening..."):
                    audio = r.listen(source, phrase_time_limit=5)
                try:
                    return r.recognize_google(audio)
                except sr.UnknownValueError:
                    return "Sorry, I couldn't understand your voice."
                except sr.RequestError:
                    return "Speech recognition service failed. Try again."
        except AttributeError:
            return "PyAudio is not installed or microphone not accessible."

    # Text-to-speech output using Streamlit audio widget
    def speak_text(text, lang_code):
        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.audio(mp3_fp, format="audio/mp3")

    # --- Added: Initialize session state variables ---
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if 'final_response' not in st.session_state:
        st.session_state.final_response = ""

    # Streamlit UI
    st.title("🛍️ Multilingual Order Chat Assistant")

    language = st.selectbox("🌐 Choose your language:", list(language_codes.keys()))
    lang_code = language_codes[language]

    col1, col2 = st.columns(2)

    # Use session state for input box to keep value
    user_input = col1.text_input("💬 Type your query here:", value=st.session_state.user_input)

    use_voice = col2.button("🎤 Speak your query")

    if use_voice:
        recognized_text = recognize_speech()
        st.session_state.user_input = recognized_text  # Save recognized text to session state
        user_input = recognized_text
        st.write(f"🗣️ You said: {recognized_text}")

    if user_input:
        # Save user input to session state
        st.session_state.user_input = user_input

        # Translate to English
        query_en = translate_to_english(user_input, lang_code)

        # Get detailed response
        response_en = handle_query_detailed_en(query_en)

        # Translate back to user language
        final_response = translate_back(response_en, lang_code)

        # Save response to session state
        st.session_state.final_response = final_response

    # Show the response if available
    if st.session_state.final_response:
        st.markdown("🧠 **Assistant Response:**")
        st.success(st.session_state.final_response)

        # Button to speak response
        if st.button("🔊 Speak Response"):
            speak_text(st.session_state.final_response, lang_code)

# --- Product Comparison Feature ---
def product_comparison():
    def seed_from_string(s):
        # Create a deterministic seed from string
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (10**8)

    def generate_mock_product_data(product_name):
        # Categories and features more detailed
        categories = {
            'phone': ['Battery', 'Camera', 'Display', 'Processor', 'Storage'],
            'tshirt': ['Fabric', 'Fit', 'Color Options', 'Durability', 'Comfort'],
            'pant': ['Material', 'Comfort', 'Style', 'Durability', 'Fit'],
            'watch': ['Battery Life', 'Style', 'Water Resistant', 'Features', 'Comfort'],
            'shoes': ['Comfort', 'Durability', 'Style', 'Grip', 'Weight']
        }
        
        lower_name = product_name.lower()
        
        # Detect category by keyword anywhere in name
        category = None
        for cat in categories.keys():
            if cat in lower_name:
                category = cat
                break
        if not category:
            category = random.choice(list(categories.keys()))
        
        features = categories[category]

        # Use deterministic seed per product to get stable results
        random.seed(seed_from_string(product_name))
        
        # Price ranges per category (example rough estimates)
        price_ranges = {
            'phone': (300, 1200),
            'tshirt': (10, 100),
            'pant': (20, 150),
            'watch': (50, 500),
            'shoes': (30, 300)
        }
        
        price_min, price_max = price_ranges.get(category, (10, 500))
        price = random.randint(price_min, price_max)
        
        # Ratings skewed more realistic
        rating = round(random.uniform(3.0, 5.0), 2)
        
        # Weighted choice helper
        def weighted_choice():
            choices = ['Poor', 'Average', 'Good', 'Excellent']
            weights = [5, 15, 40, 40]  # more chance of Good+ to keep quality reasonable
            return random.choices(choices, weights=weights, k=1)[0]
        
        feature_scores = {f: weighted_choice() for f in features}
        
        return {
            'name': product_name,
            'category': category,
            'price': price,
            'rating': rating,
            'features': feature_scores
        }

    def generate_comparison(products):
        output = "### Product Comparison:\n\n"
        output += f"Compared {len(products)} products:\n\n"
        
        for p in products:
            output += f"**{p['name']}** (Category: {p['category'].title()}):\n"
            output += f"- Price: ${p['price']}\n"
            output += f"- Rating: {p['rating']} / 5\n"
            output += "- Features:\n"
            for f, score in p['features'].items():
                output += f"  - {f}: {score}\n"
            output += "\n"
        
        cheapest = min(products, key=lambda x: x['price'])
        best_rated = max(products, key=lambda x: x['rating'])
        
        output += f"**Cheapest product:** {cheapest['name']} at ${cheapest['price']}.\n"
        output += f"**Best rated product:** {best_rated['name']} with rating {best_rated['rating']}.\n\n"
        
        output += "Feature Highlights:\n"
        all_features = set()
        for p in products:
            all_features.update(p['features'].keys())
        all_features = sorted(all_features)
        
        order = {'Poor':1, 'Average':2, 'Good':3, 'Excellent':4}
        for f in all_features:
            scores = []
            for p in products:
                score = p['features'].get(f, 'N/A')
                scores.append((p['name'], score))
            # ignore N/A in sorting
            filtered = [s for s in scores if s[1] != 'N/A']
            if not filtered:
                continue
            filtered.sort(key=lambda x: order.get(x[1], 0), reverse=True)
            best = filtered[0]
            output += f"- Best {f}: {best[0]} ({best[1]})\n"
        
        return output

    def main():
        st.title("Product Comparison")
        st.write("Enter product names (any products, comma separated). Example: iPhone 13, Nike shoes, Levi's pant")
        
        product_input = st.text_input("Enter products:")
        if product_input:
            product_names = [p.strip() for p in product_input.split(',') if p.strip()]
            
            if len(product_names) < 2:
                st.warning("Please enter at least 2 products for comparison.")
                return
            
            products = [generate_mock_product_data(name) for name in product_names]
            comparison_text = generate_comparison(products)
            st.markdown(comparison_text)

    main()

# --- Inventory Management Feature ---
def inventory_management():
    @st.cache_data
    def load_sample_data():
        products = [
            {"product_id": 1, "name": "Sneakers", "category": "Footwear", "price": 70},
            {"product_id": 2, "name": "Sandals", "category": "Footwear", "price": 30},
            {"product_id": 3, "name": "T-shirt", "category": "Apparel", "price": 25},
            {"product_id": 4, "name": "Jeans", "category": "Apparel", "price": 50},
            {"product_id": 5, "name": "Hat", "category": "Accessories", "price": 20},
            {"product_id": 6, "name": "Backpack", "category": "Accessories", "price": 80},
            {"product_id": 7, "name": "Running Shorts", "category": "Apparel", "price": 35},
        ]
        products_df = pd.DataFrame(products)

        np.random.seed(42)
        sales = [{"product_id": p["product_id"], "units_sold": np.random.randint(50, 200)} for p in products]
        views = [{"product_id": p["product_id"], "views": np.random.randint(1000, 5000)} for p in products]
        inventory = [{"product_id": p["product_id"], "stock": np.random.randint(5, 50)} for p in products]

        df = products_df.merge(pd.DataFrame(sales), on="product_id") \
                        .merge(pd.DataFrame(views), on="product_id") \
                        .merge(pd.DataFrame(inventory), on="product_id")
        return df

    def top_selling_products(df, top_n=5):
        return df.sort_values("units_sold", ascending=False).head(top_n)

    def most_viewed_products(df, top_n=5):
        return df.sort_values("views", ascending=False).head(top_n)

    def low_stock_alerts(df, threshold=10):
        return df[df["stock"] < threshold]

    def category_wise_rankings(df):
        df["category_rank"] = df.groupby("category")["units_sold"].rank(ascending=False, method="first")
        return df.sort_values(["category", "category_rank"])

    def convert_to_excel(tables: dict) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for sheet_name, table_df in tables.items():
                table_df.to_excel(writer, index=False, sheet_name=sheet_name)
        return output.getvalue()

    def main():
        st.title("📊 Inventory Dashboard")

        df = load_sample_data()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Top 5 Selling Products")
            top_selling = top_selling_products(df)
            st.table(top_selling[["name", "units_sold"]])

            st.subheader("⚠️ Low Stock Alerts")
            low_stock = low_stock_alerts(df)
            if low_stock.empty:
                st.success("✅ All items are sufficiently stocked.")
            else:
                st.table(low_stock[["name", "stock"]])

        with col2:
            st.subheader("🔥 Most Viewed Products")
            most_viewed = most_viewed_products(df)
            st.table(most_viewed[["name", "views"]])

            st.subheader("🏆 Category-wise Rankings")
            category_ranks = category_wise_rankings(df)
            st.dataframe(category_ranks[["category", "name", "units_sold", "category_rank"]])

        st.divider()
        st.subheader("⬇️ Download All Tables as Excel")

        # Prepare export tables
        export_data = {
            "Top Selling": top_selling,
            "Low Stock": low_stock,
            "Most Viewed": most_viewed,
            "Category Rankings": category_ranks,
            "Full Inventory": df
        }

        excel_file = convert_to_excel(export_data)
        st.download_button("📥 Download Excel Report", excel_file, "sales_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    main()

# --- Main App Navigation ---
def main():
    st.set_page_config(page_title="E-Commerce Assistant App", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a feature:", [
        "Product Search",
        "Image-Based Product Finder",
        "Multilingual Chatbot Assistance",
        "Product Comparison",
        "Inventory Management"
    ])

    if page == "Product Search":
        product_search()
    elif page == "Image-Based Product Finder":
        image_description()
    elif page == "Multilingual Chatbot Assistance":
        multilingual_chatbot_assistance()
    elif page == "Product Comparison":
        product_comparison()
    elif page == "Inventory Management":
        inventory_management()

if __name__ == "__main__":
    main()
