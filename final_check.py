import app
print("App imports successfully")
indexed = app.load_indexed_md()
print(f"Loaded {len(indexed["sections"])} sections from document_clean.md")
print(f"Introduction content length: {len(app.get_introduction_content(indexed["md_text"]))} chars")
print("Ready for Streamlit Community Cloud deployment")
