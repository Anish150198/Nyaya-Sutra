import sys
sys.path.insert(0, '.')
import chromadb
client = chromadb.PersistentClient(path='./data/gold/chromadb')
for col in client.list_collections():
    print(f'Deleting collection: {col.name} ({col.count()} vectors)')
    client.delete_collection(col.name)
print('All old collections deleted.')

