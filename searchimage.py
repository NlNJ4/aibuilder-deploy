import requests
from PIL import Image
from io import BytesIO
from IPython.display import display

def google_image_search(query,num_images=1):
    search_url = "https://www.googleapis.com/customsearch/v1"
    api_key = 'AIzaSyCBo9htNC3hgUF_gVZlYqA1Y-3SF6j8gYQ'
    cx = '4572b0e32e6af4d2c'
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

def display_images(image_urls, size=(200, 200)):
    for url in image_urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            display(img)
        except Exception as e:
            print(f"Could not load image from {url} - {e}")

# Replace with your actual API key and CX
query = 'cats'

image_urls = google_image_search(query)
display_images(image_urls)