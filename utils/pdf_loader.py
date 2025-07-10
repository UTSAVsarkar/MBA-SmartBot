from PyPDF2 import PdfReader

def load_and_chunk_pdfs(file_paths, chunk_size=500, overlap=50):
    chunks = []
    for path in file_paths:
        reader = PdfReader(path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks
