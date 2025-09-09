from pdf_hybrid_utility import HybridPDFProcessor

# Initialize processor
processor = HybridPDFProcessor(
    text_threshold=50,      # Min characters for text-based detection
    confidence_threshold=60 # Min OCR confidence
)

# Process a PDF
results = processor.process_pdf("input_pdfs/testing.pdf", "output_folder2")

# Check results
print(f"Text-based pages: {results['processing_summary']['text_based_pages']}")
print(f"Image-based pages: {results['processing_summary']['image_based_pages']}")