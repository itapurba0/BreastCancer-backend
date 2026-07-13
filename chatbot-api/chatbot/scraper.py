import os
import requests
from bs4 import BeautifulSoup

# Define where to save the text
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def scrape_who_breast_cancer_facts():
    """Scrapes the official WHO fact sheet on breast cancer."""
    
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    url = "https://www.who.int/news-room/fact-sheets/detail/breast-cancer"
    print(f"Fetching data from WHO: {url}...")
    
    try:
        # Use a standard user-agent so the website doesn't block us
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # The WHO article content is typically held within <article> tags or main div blocks.
        # We will extract all paragraph <p> tags, which contain the core medical text.
        paragraphs = soup.find_all('p')
        
        # Clean and combine the text
        extracted_text = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Filter out short UI elements or navigation junk
            if len(text) > 50: 
                extracted_text.append(text)
                
        final_document = "\n\n".join(extracted_text)
        
        if not final_document:
            print("Failed to extract text. The WHO website layout may have changed.")
            return

        # Save to your data folder
        file_path = os.path.join(DATA_DIR, "who_breast_cancer_facts.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(final_document)
            
        print(f"Success! Scraped {len(extracted_text)} paragraphs of medical data.")
        print(f"Saved to: {file_path}")

    except Exception as e:
        print(f"An error occurred during scraping: {e}")

if __name__ == "__main__":
    scrape_who_breast_cancer_facts()