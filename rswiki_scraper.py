import requests
import os
import gzip
from lxml import etree
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma


load_dotenv()


class RswikiScraper:
    def __init__(self,main_sitemap="https://runescape.wiki/images/sitemaps/index.xml",delay_s=1,db_loc="./chroma.db"):
        self.main_sitemap = main_sitemap
        self.delay_s = delay_s
        self.webpages = []
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.chroma_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=db_loc,  # directory to store the vector store
            collection_name="runescape",  # name of the collection
        )

    def download_sitemaps(self):
        # Return without processing for now
        sitemaps =  ["https://runescape.wiki/images/sitemaps/NS_0-0.xml.gz"]
        for sitemap in sitemaps:
            time.sleep(self.delay_s)
            self.download_sitemap(sitemap)

    def process_webpages(self):
        # Return without processing for now
        for webpage in self.webpages:
            time.sleep(self.delay_s)
            self.process_webpage(webpage)

    def db_add_via_similarity_check(self,all_splits):
        # Avoid adding duplicate documents by checking for high similarity
        for split_doc in all_splits:
            results = self.chroma_db.similarity_search_with_score(
                split_doc.page_content, k=1
            )
            # Try to get score if available
            score = None
            if results:
                # If result is tuple (doc, score)
                if isinstance(results[0], tuple) and len(results[0]) > 1:
                    score = results[0][1]
            # Only add if no similar doc or similarity < 0.99
            if not results or (score is not None and score > 0.01):
                self.chroma_db.add_documents([split_doc])
            else:
                print("Skipped adding a highly similar document (>99%).")        

    def add_to_vector_db(self,docs,webpage):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        results = self.chroma_db._collection.get(where={"source": webpage})
        print(len(results['ids']))
        if len(results['ids']) != len(all_splits):
            if len(results['ids']) != 0:
                self.chroma_db._collection.delete(ids=results['ids'])
            self.chroma_db.add_documents(all_splits)
        else:
            print("Already added to database before")    
        # self.db_add_via_similarity_check(all_splits)


    def process_webpage(self,webpage):
        response = requests.get(webpage)
        if response.status_code == 200:
            loader = WebBaseLoader(
                web_paths=(webpage,),
            )
            docs = loader.load()
            self.add_to_vector_db(docs,webpage)
            # print(f"Wrote {webpage} to chroma")             

        else:
            print(f"Failed to download. Status code: {response.status_code}")

    def process_sitemaps(self):
        for file in os.listdir("./sitemaps"):
            full_path = os.path.join("./sitemaps", file)
            print(full_path)
            with gzip.open(full_path, 'rb') as f:
                xml_bytes = f.read()
                root = etree.fromstring(xml_bytes)
                print(root)
                ns = {'sm':root.nsmap.get(None)}
                print("Default namespace:", ns)

                # Use XPath with namespace prefix to find all <loc> elements
                loc_elements = root.xpath('//sm:loc', namespaces=ns)

                # Print the text inside each <loc>
                for loc in loc_elements:
                    self.webpages.append(loc.text)
        print(len(self.webpages))           
    
    def download_sitemap(self,sitemap):
        gz_filepath = os.path.join("./sitemaps/",os.path.basename(sitemap))
        response = requests.get(sitemap)
        if response.status_code == 200:
            with open(gz_filepath, "wb") as f:
                f.write(response.content)
            print(f"Compressed file saved to: {gz_filepath}")             

        else:
            print(f"Failed to download. Status code: {response.status_code}")



def main():
    print("Hello from runescapewikibot!")
    rsws = RswikiScraper()
    # rsws.download_sitemaps()
    rsws.process_sitemaps()
    rsws.process_webpages()
    # rsws.process_webpage("https://runescape.wiki/w/List_of_quests")


if __name__ == "__main__":
    main()
