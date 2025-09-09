# PDF Hybrid Processing Utility

A comprehensive Python utility for processing mixed PDF documents with automatic page type detection and intelligent routing to appropriate extraction pipelines.

## 🚀 Features

- **Smart Page Detection**: Automatically determines if pages are text-based or image-based
- **Dual Processing Pipelines**: 
  - Text-based extraction for digital PDFs (fast and accurate)
  - OCR-based extraction for scanned/image PDFs (comprehensive)
- **Unified Output Format**: Consistent JSON structure regardless of processing method
- **Content Order Preservation**: Maintains proper reading order in mixed documents
- **Quality Metrics**: Confidence scores and extraction quality assessment
- **Batch Processing**: Handle multiple PDFs efficiently
- **Configurable Thresholds**: Customize detection and quality parameters

##  Installation

### Prerequisites

1. **Python 3.7+**
2. **Tesseract OCR** (for image-based processing)
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Default Windows path: `C:\Users\[username]\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install PyMuPDF opencv-python Pillow pytesseract pandas numpy matplotlib tqdm
```

##  Package Structure

```
pdf_hybrid_utility/
├── __init__.py              # Package initialization
├── core/
│   ├── __init__.py
│   └── hybrid_processor.py  # Main hybrid processor
├── processors/
│   ├── __init__.py
│   ├── text_processor.py    # Text-based PDF processing
│   └── image_processor.py   # Image-based PDF processing
├── utils/
│   ├── __init__.py
│   ├── detector.py          # Page type detection
│   └── metrics.py           # Quality metrics calculation
├── examples/
│   ├── basic_usage.py       # Basic usage examples
│   └── advanced_usage.py    # Advanced patterns
├── requirements.txt         # Dependencies
└── README.md               # This file
```

##  Quick Start

### Basic Usage

```python
from pdf_hybrid_utility import HybridPDFProcessor

# Initialize processor
processor = HybridPDFProcessor(
    text_threshold=50,      # Min characters for text-based detection
    confidence_threshold=70 # Min OCR confidence
)

# Process a PDF
results = processor.process_pdf("document.pdf", "output_folder")

# Check results
print(f"Text-based pages: {results['processing_summary']['text_based_pages']}")
print(f"Image-based pages: {results['processing_summary']['image_based_pages']}")
```

### Single Page Processing

```python
# Process just one page
page_result = processor.process_single_page("document.pdf", page_number=1)

if page_result['success']:
    print(f"Processing method: {page_result['processing_method']}")
    print(f"Content items: {len(page_result['content'])}")
```

### Pre-Analysis

```python
# Analyze document composition before processing
analysis = processor.analyze_pdf_composition("document.pdf")

print(f"Text pages: {analysis['text_based_pages']}")
print(f"Image pages: {analysis['image_based_pages']}")
print(f"Recommended strategy: {analysis['strategy']}")
```

##  Detailed Usage

### Configuration Options

```python
# High precision (strict thresholds)
processor = HybridPDFProcessor(
    text_threshold=100,     # Only clear text pages → text pipeline
    confidence_threshold=85 # Only high-quality OCR results
)

# High recall (lenient thresholds)
processor = HybridPDFProcessor(
    text_threshold=10,      # Catch minimal text → text pipeline
    confidence_threshold=30 # Accept more OCR results
)
```

### Batch Processing

```python
from pathlib import Path

pdf_files = list(Path("input_folder").glob("*.pdf"))

for pdf_path in pdf_files:
    results = processor.process_pdf(str(pdf_path), "batch_output")
    print(f"Processed {pdf_path.name}: {results['processing_summary']}")
```

### Custom Processing Logic

```python
# Page-by-page with custom logic
for page_num in range(1, total_pages + 1):
    page_result = processor.process_single_page(pdf_path, page_num)
    
    if page_result['success']:
        # Custom business logic
        if 'table' in [item['type'] for item in page_result['content']]:
            print(f"Page {page_num}: Contains tables")
            # Handle table extraction specifically
        
        if page_result['metrics']['avg_confidence'] < 0.7:
            print(f"Page {page_num}: Low confidence - manual review needed")
```

##  Output Format

The utility produces a standardized JSON structure:

```json
{
  "pdf_file": "document.pdf",
  "total_pages": 10,
  "processing_summary": {
    "text_based_pages": 7,
    "image_based_pages": 3,
    "failed_pages": 0,
    "total_content_items": 45
  },
  "pages": [
    {
      "page_number": 1,
      "processing_method": "text_based",
      "content": [
        {
          "type": "text",
          "content": "Document text...",
          "bbox_original": [x1, y1, x2, y2],
          "bbox_normalized": [0.1, 0.2, 0.8, 0.3],
          "confidence": 1.0,
          "processing_method": "text_extraction"
        }
      ],
      "metrics": {
        "quality_score": 95.0,
        "accuracy_percentage": 95.0,
        "avg_confidence": 0.95
      },
      "success": true
    }
  ]
}
```

##  Page Detection Logic

The utility uses a multi-step approach:

1. **Text Extraction Test**: Uses PyMuPDF's `get_text()` to extract text
2. **Character Count Check**: Counts extractable characters vs threshold  
3. **Text Block Analysis**: Checks for structured text elements
4. **Image Coverage Analysis**: Estimates image vs text content ratio

**Decision Logic:**
- **Text-based**: If extractable text ≥ threshold AND structured text blocks exist
- **Image-based**: Otherwise (scanned, image-heavy, or corrupted text)

##  Performance Tuning

### Optimization Tips

- **For mostly text PDFs**: Increase `text_threshold` to avoid unnecessary OCR
- **For mostly scanned PDFs**: Decrease `text_threshold` for faster OCR routing  
- **For large documents**: Use single-page processing to manage memory
- **For batch processing**: Use lower confidence thresholds to speed up OCR

### Performance Comparison

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Text-only | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Digital PDFs |
| OCR-only | ⚡ | ⭐⭐⭐ | Scanned documents |
| Hybrid | ⚡⚡ | ⭐⭐⭐⭐ | Mixed documents |

## 🔍 Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Install Tesseract OCR
   - Update path in `processors/image_processor.py`

2. **"OpenCV not available"**
   - Install: `pip install opencv-python`

3. **Poor OCR results**
   - Increase DPI in image processing
   - Adjust preprocessing parameters
   - Lower confidence threshold

4. **Memory issues with large PDFs**
   - Use page-by-page processing
   - Reduce image resolution for OCR

### Debug Mode

```python
# Check processing capabilities
capabilities = processor.get_processing_capabilities()
print(capabilities)

# Analyze before processing
analysis = processor.analyze_pdf_composition("document.pdf")
print(f"Recommended strategy: {analysis['strategy']}")
```

##  Examples

### Run Basic Examples
```bash
cd examples
python basic_usage.py
```

### Run Advanced Examples  
```bash
cd examples
python advanced_usage.py
```

##  Integration Patterns

### 1. Pipeline Integration
```python
# Pre-analysis → Processing → Post-processing
analysis = processor.analyze_pdf_composition(pdf_path)
results = processor.process_pdf(pdf_path, output_dir)
summary = MetricsCalculator.create_processing_summary(results)
```

### 2. Error Handling
```python
try:
    results = processor.process_pdf(pdf_path)
    if results['processing_summary']['failed_pages'] > 0:
        # Handle failed pages
        pass
except Exception as e:
    # Handle processing errors
    pass
```

### 3. Quality Assurance
```python
for page in results['pages']:
    if page['success']:
        quality = page['metrics']['quality_score']
        if quality < 70:
            # Flag for manual review
            pass
```

##  License

This project is licensed under the MIT License.

##  Acknowledgments

- Built on PyMuPDF, OpenCV, and Tesseract OCR
- Inspired by the need for robust mixed-document processing
- Thanks to the open-source computer vision and OCR communities
