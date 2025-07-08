import streamlit as st
import random
import hashlib

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
    st.title("Improved Dynamic Product Comparison")
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

if __name__ == "__main__":
    main()
