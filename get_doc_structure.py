from docx import Document
import sys

def get_structure(file_path):
    try:
        doc = Document(file_path)
        print("--- HEADERS ---")
        for Paragraph in doc.paragraphs:
            if Paragraph.style.name.startswith('Heading'):
                print(f"{Paragraph.style.name}: {Paragraph.text}")
            elif Paragraph.text.isupper() and len(Paragraph.text) < 100: # Heuristic for non-styled headers
                 print(f"POTENTIAL HEADER: {Paragraph.text}")
        print("--- END HEADERS ---")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_structure(sys.argv[1])
