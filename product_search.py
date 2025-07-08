import streamlit as st
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyC8Qamg9uhd-jfUtMQ1fbyZ3CXwarfziKA")  # 🔑 Replace with your actual key

# Load Gemini 1.5 Flash model
model = genai.GenerativeModel("models/gemini-1.5-flash")

st.title("🛍️ GenAI Product Finder (Gemini Powered)")

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
