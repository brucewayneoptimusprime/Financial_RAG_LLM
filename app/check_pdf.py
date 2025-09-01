# app/check_pdf.py
from pathlib import Path
import pdfplumber

# 1) point to your file
PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "filings" / "Apple_10K_2023.pdf"

def main():
    if not PDF_PATH.exists():
        print(f"❌ File not found:\n{PDF_PATH}\n"
              "Make sure the filename matches exactly (case matters).")
        return

    with pdfplumber.open(PDF_PATH) as pdf:
        n_pages = len(pdf.pages)
        print(f"Opened PDF: {PDF_PATH.name}")
        print(f"   Pages: {n_pages}")

        # 2) show a short preview from the first page
        first_page = pdf.pages[0]
        text = (first_page.extract_text() or "").strip()
        preview = text[:800].replace("\n", " ")
        print("\n--- Page 1 preview (first ~800 chars) ---")
        print(preview if preview else "(no extractable text on page 1)")
        print("-----------------------------------------")

if __name__ == "__main__":
    main()
