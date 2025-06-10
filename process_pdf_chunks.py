import os
from PyPDF2 import PdfReader
from typing import List, Dict

def process_pdfs(pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Process PDF files for RAG applications with metadata preservation.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        chunk_size (int): Character length for text chunks
        chunk_overlap (int): Overlap between consecutive chunks
    
    Returns:
        List of dictionaries with text chunks and metadata
    """
    documents = []
    
    # Get all PDF files in directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                doc_text = ""
                meta = {
                    'source': pdf_file,
                    'total_pages': len(reader.pages),
                    'pages': []
                }
                
                # Extract text with page numbers
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        doc_text += f"Page {page_num+1}:\n{page_text}\n"
                        meta['pages'].append(page_num+1)
                
                # Split document into chunks
                chunks = []
                start = 0
                while start < len(doc_text):
                    end = min(start + chunk_size, len(doc_text))
                    chunk = doc_text[start:end].strip()
                    
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            **meta,
                            'chunk_id': f"{pdf_file}-{len(chunks)+1}",
                            'start_char': start,
                            'end_char': end
                        }
                    })
                    
                    start += chunk_size - chunk_overlap
                
                documents.extend(chunks)
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue

    return documents

# Example usage
if __name__ == "__main__":
    # Configuration
    PDF_DIRECTORY = "Compliance_pdfs"  # Directory with your PDFs
    OUTPUT_FILE = "processed_chunks.json"
    
    # Process documents
    processed_data = process_pdfs(PDF_DIRECTORY)
    
    # Save results
    import json
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} chunks from {len(set(d['metadata']['source'] for d in processed_data))} PDFs")
