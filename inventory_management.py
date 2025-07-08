import streamlit as st
import pandas as pd
import numpy as np
import io

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
    st.set_page_config(layout="wide")
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

if __name__ == "__main__":
    main()
