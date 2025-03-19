import pysolr

# Connect to the Solr core for ev_opinions
solr = pysolr.Solr('http://localhost:8983/solr/ev_opinions', always_commit=False)

# Query for all documents. Adjust 'rows' to control the number of documents retrieved.
results = solr.search('*:*', rows=100)

# Iterate and print each document
if __name__ == "__main__":
    for doc in results:
        print(doc)