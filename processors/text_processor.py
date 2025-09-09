"""
Text-Based PDF Processor
Handles text-based PDF pages using PyMuPDF for direct text extraction
"""

import sys
import os
import fitz

# Add parent directory to path to import layout_lm1
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from layout_lm1 import PDFExtractor
    TEXT_EXTRACTOR_AVAILABLE = True
except ImportError:
    print("Warning: layout_lm1.py not found. Text-based processing may be limited.")
    TEXT_EXTRACTOR_AVAILABLE = False

from ..utils.metrics import MetricsCalculator


class TextBasedProcessor:
    """
    Processes text-based PDF pages using direct text extraction
    """
    
    def __init__(self):
        """Initialize the text-based processor"""
        if TEXT_EXTRACTOR_AVAILABLE:
            self.extractor = PDFExtractor()
        else:
            self.extractor = None
    
    def process_page(self, pdf_path, page_number):
        """
        Process a text-based PDF page
        
        Args:
            pdf_path (str): Path to PDF file
            page_number (int): Page number (1-based)
            
        Returns:
            dict: Processing results
        """
        if not self.extractor:
            return self._fallback_text_processing(pdf_path, page_number)
        
        print(f"   → Processing page {page_number} using TEXT-BASED pipeline")
        
        try:
            # Convert to 0-based indexing
            page_num = page_number - 1
            
            # Extract text blocks directly first (for entity extraction and metrics)
            raw_text_blocks = self.extractor.extract_text_blocks(pdf_path, page_num)
            
            # Extract content with positions to preserve order
            content_items = self.extractor.extract_content_with_positions(pdf_path, page_num)
            
            page_content = []
            page_tables = []
            page_images = []
            page_text_blocks = raw_text_blocks  # Use raw text blocks for entity extraction
            
            if content_items:
                # Process content items in order
                for item in content_items:
                    if item['type'] == 'text':
                        text_block = {
                            "type": "text",
                            "content": item['content'].strip(),
                            "bbox_original": item['bbox'],
                            "bbox_normalized": item['bbox_normalized'],
                            "confidence": 1.0,  # Text extraction from PDF has high confidence
                            "processing_method": "text_extraction"
                        }
                        page_content.append(text_block)
                        
                    elif item['type'] == 'table':
                        table_block = {
                            "type": "table",
                            "content": item['content'],
                            "bbox_original": item['bbox'],
                            "bbox_normalized": item['bbox_normalized'],
                            "confidence": item.get('confidence', 0.9),
                            "processing_method": "text_extraction"
                        }
                        page_content.append(table_block)
                        page_tables.append(table_block)
                        
                    elif item['type'] == 'image':
                        image_block = {
                            "type": "image",
                            "content": item['content'],
                            "bbox_original": item['bbox'],
                            "bbox_normalized": item['bbox_normalized'],
                            "confidence": 1.0,
                            "processing_method": "text_extraction"
                        }
                        page_content.append(image_block)
                        page_images.append(image_block)
            
            # Extract entities using raw text blocks (they have the correct 'text' key)
            entities = []
            if page_text_blocks:
                entities = self.extractor.extract_entities_simple(page_text_blocks)
            
            # Calculate metrics using text extractor's method
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            image_size = (int(page.rect.width), int(page.rect.height))
            doc.close()
            
            metrics = MetricsCalculator.calculate_text_based_metrics(
                page_text_blocks, entities, page_tables, image_size
            )
            
            return {
                'page_number': page_number,
                'processing_method': 'text_based',
                'content': page_content,
                'tables': page_tables,
                'images': page_images,
                'text_blocks': page_text_blocks,
                'entities': entities,
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Error in text-based processing: {e}")
            return {
                'page_number': page_number,
                'processing_method': 'text_based',
                'error': str(e),
                'success': False
            }
    
    def _fallback_text_processing(self, pdf_path, page_number):
        """
        Fallback text processing when layout_lm1 is not available
        """
        print(f"   → Processing page {page_number} using FALLBACK text extraction")
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            
            # Simple text extraction
            text_content = page.get_text()
            text_dict = page.get_text("dict")
            
            # Basic text blocks extraction
            text_blocks = []
            page_content = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            block_text += span.get("text", "") + " "
                    
                    if block_text.strip():
                        bbox = block.get("bbox", [0, 0, 0, 0])
                        text_block = {
                            "text": block_text.strip(),
                            "bbox": bbox
                        }
                        text_blocks.append(text_block)
                        
                        content_block = {
                            "type": "text",
                            "content": block_text.strip(),
                            "bbox_original": bbox,
                            "bbox_normalized": self._normalize_bbox(bbox, (page.rect.width, page.rect.height)),
                            "confidence": 1.0,
                            "processing_method": "fallback_text_extraction"
                        }
                        page_content.append(content_block)
            
            doc.close()
            
            # Basic metrics
            metrics = {
                'total_text_blocks': len(text_blocks),
                'total_entities': 0,
                'total_tables': 0,
                'total_text_length': len(text_content),
                'avg_confidence': 1.0,
                'quality_score': 80.0 if text_blocks else 0.0,
                'accuracy_percentage': 80.0 if text_blocks else 0.0,
                'processing_method': 'fallback_text_extraction'
            }
            
            return {
                'page_number': page_number,
                'processing_method': 'text_based_fallback',
                'content': page_content,
                'tables': [],
                'images': [],
                'text_blocks': text_blocks,
                'entities': [],
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Error in fallback text processing: {e}")
            return {
                'page_number': page_number,
                'processing_method': 'text_based_fallback',
                'error': str(e),
                'success': False
            }
    
    def _normalize_bbox(self, bbox, page_size):
        """Normalize bounding box coordinates to 0-1 range"""
        width, height = page_size
        x1, y1, x2, y2 = bbox
        return [x1/width, y1/height, x2/width, y2/height]
