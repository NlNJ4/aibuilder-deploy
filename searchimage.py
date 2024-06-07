import requests
from PIL import Image
from io import BytesIO
import streamlit as st

def google_image_search(query, num_images=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    api_key = 'AIzaSyCBo9htNC3hgUF_gVZlYqA1Y-3SF6j8gYQ'  # Replace with your actual API key
    cx = '4572b0e32e6af4d2c'            # Replace with your Custom Search Engine ID
    params = {
        'q': query,
        'key': api_key,
        'cx': cx,
        'searchType': 'image',
        'num': num_images
    }

    response = requests.get(search_url, params=params)
    results = response.json()

    image_urls = []
    if 'items' in results:
        for item in results['items']:
            image_urls.append(item['link'])
    
    return image_urls

def display_images(image_urls, size=(800, 800)):
    image_urls = image_urls[1:]
    for url in image_urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = img.resize(size)
            st.image(img, caption=url, use_column_width=True)
            return 0
        except Exception as e:
            continue
            #st.write(f"Could not load image from {url} - {e}")