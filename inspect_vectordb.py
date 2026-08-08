import yaml
from colorama import Fore, Style, init
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

init(autoreset=True)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name=config['embeddings']['model_name'])
vectorstore = FAISS.load_local(
    config['vectordb']['persist_directory'],
    embeddings,
    allow_dangerous_deserialization=True  # safe: loading our own local index
)

print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
print(f"{Fore.CYAN}🔍 FAISS DATABASE INSPECTOR{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

# Get index info
all_docs = list(vectorstore.docstore._dict.values())
print(f"{Fore.YELLOW}📊 Index Stats:{Style.RESET_ALL}")
print(f"   • Total vectors: {vectorstore.index.ntotal}")
print(f"   • Embedding dimension: {vectorstore.index.d}")
print(f"   • Storage path: {config['vectordb']['persist_directory']}\n")

# Sample some documents
print(f"{Fore.YELLOW}📄 Sample Documents (first 5):{Style.RESET_ALL}\n")

for i, doc in enumerate(all_docs[:5], 1):
    print(f"{Fore.CYAN}Document {i}:{Style.RESET_ALL}")
    print(f"   Source: {doc.metadata.get('source', 'N/A')}")
    print(f"   Page: {doc.metadata.get('page', 'N/A')}")
    print(f"   Content preview: {doc.page_content[:150]}...")
    print()

# Show unique sources
sources = set(doc.metadata.get('source', 'Unknown') for doc in all_docs)

print(f"{Fore.YELLOW}📚 Unique Source Documents:{Style.RESET_ALL}")
for source in sorted(sources):
    source_count = sum(1 for doc in all_docs if doc.metadata.get('source') == source)
    print(f"   • {source}: {source_count} chunks")

print()
