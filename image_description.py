import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# Configure Gemini API Key
genai.configure(api_key="AIzaSyC8Qamg9uhd-jfUtMQ1fbyZ3CXwarfziKA")  # Replace with your actual key

# Load the Gemini 1.5 Flash model
model = genai.GenerativeModel("models/gemini-1.5-flash")

st.set_page_config(page_title="🖼️ Image-Based Product Finder")
st.title("🖼️ GenAI Image-Based Product Finder")

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
        st.markdown(f"[🛒 Buy Now]({product['link']})", unsafe_allow_html=True)
    else:
        st.error("❌ Unable to parse product recommendation details. Please try again.")
else:
    st.info("📷 Upload an image to start.")
