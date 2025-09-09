"""
Image-Based PDF Processor
Handles scanned/image-based PDF pages using OCR and computer vision
"""

import fitz
import numpy as np
from collections import defaultdict
from io import BytesIO

# Image processing imports
try:
    import cv2 as cv
    import pytesseract
    from PIL import Image
    OPENCV_AVAILABLE = True
    
    # Configure Tesseract path for Windows
    try:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ntallapalli\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    except:
        pass
        
except ImportError:
    print("Warning: OpenCV/PIL/Tesseract not available. Image-based processing will be disabled.")
    OPENCV_AVAILABLE = False

from ..utils.metrics import MetricsCalculator


class ImageBasedProcessor:
    """
    Processes image-based PDF pages using OCR and computer vision techniques
    """
    
    def __init__(self, confidence_threshold=70):
        """
        Initialize the image-based processor
        
        Args:
            confidence_threshold (int): Minimum OCR confidence for acceptance
        """
        self.confidence_threshold = confidence_threshold / 100.0  # Convert to 0-1 scale
        
        if not OPENCV_AVAILABLE:
            print("Warning: Image processing dependencies not available")
    
    def process_page(self, pdf_path, page_number):
        """
        Process an image-based PDF page using OCR
        
        Args:
            pdf_path (str): Path to PDF file
            page_number (int): Page number (1-based)
            
        Returns:
            dict: Processing results
        """
        if not OPENCV_AVAILABLE:
            return {
                'page_number': page_number,
                'processing_method': 'image_based',
                'error': 'OpenCV/Tesseract not available',
                'success': False
            }
        
        print(f"   → Processing page {page_number} using IMAGE-BASED pipeline")
        
        try:
            # Convert PDF page to image
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            
            # Convert page to image with high DPI
            mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            pil_image = Image.open(BytesIO(img_data))
            doc.close()
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image_for_ocr(pil_image)
            
            # Extract text using OCR
            text_data = self._extract_text_with_ocr(processed_image)
            
            # Detect and extract tables
            tables = self._detect_and_extract_tables(processed_image, pil_image)
            
            # Organize content by position (top to bottom)
            page_content = []
            
            # Add text blocks
            for text_block in text_data:
                if text_block['confidence'] >= self.confidence_threshold:
                    content_block = {
                        "type": "text",
                        "content": text_block['text'],
                        "bbox_original": text_block['bbox'],
                        "bbox_normalized": self._normalize_bbox(text_block['bbox'], pil_image.size),
                        "confidence": text_block['confidence'],
                        "processing_method": "ocr"
                    }
                    page_content.append(content_block)
            
            # Add tables
            for table in tables:
                content_block = {
                    "type": "table",
                    "content": table['content'],
                    "bbox_original": table['bbox'],
                    "bbox_normalized": self._normalize_bbox(table['bbox'], pil_image.size),
                    "confidence": table['confidence'],
                    "processing_method": "ocr_table_detection"
                }
                page_content.append(content_block)
            
            # Sort content by y-position (top to bottom)
            page_content.sort(key=lambda x: x['bbox_original'][1])
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_image_based_metrics(page_content)
            
            return {
                'page_number': page_number,
                'processing_method': 'image_based',
                'content': page_content,
                'tables': [b for b in page_content if b['type'] == 'table'],
                'text_blocks': [b for b in page_content if b['type'] == 'text'],
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Error in image-based processing: {e}")
            return {
                'page_number': page_number,
                'processing_method': 'image_based',
                'error': str(e),
                'success': False
            }
    
    def _preprocess_image_for_ocr(self, image):
        """
        Fast image preprocessing for OCR
        """
        # Convert to grayscale
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        # Convert to numpy for processing
        gray_array = np.array(gray_image)
        
        # Apply binary thresholding
        _, binary_array = cv.threshold(gray_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Convert back to PIL
        return Image.fromarray(binary_array)
    
    def _extract_text_with_ocr(self, image):
        """
        Extract text blocks using OCR
        """
        # Configure OCR
        config = r'--oem 3 --psm 6'
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        
        # Group words into text blocks
        blocks = defaultdict(list)
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                block_num = data['block_num'][i]
                blocks[block_num].append({
                    'text': data['text'][i],
                    'conf': int(data['conf'][i]),
                    'x': int(data['left'][i]),
                    'y': int(data['top'][i]),
                    'w': int(data['width'][i]),
                    'h': int(data['height'][i])
                })
        
        # Combine words into text blocks
        text_blocks = []
        for block_num, words in blocks.items():
            if len(words) > 0:
                # Combine text
                block_text = ' '.join([word['text'] for word in words])
                
                # Calculate bounding box
                min_x = min([word['x'] for word in words])
                min_y = min([word['y'] for word in words])
                max_x = max([word['x'] + word['w'] for word in words])
                max_y = max([word['y'] + word['h'] for word in words])
                
                # Calculate average confidence
                avg_conf = np.mean([word['conf'] for word in words])
                
                if len(block_text.strip()) > 5:  # Only include substantial text blocks
                    text_blocks.append({
                        'text': block_text.strip(),
                        'bbox': [min_x, min_y, max_x, max_y],
                        'confidence': avg_conf / 100.0  # Convert to 0-1 scale
                    })
        
        return text_blocks
    
    def _detect_and_extract_tables(self, processed_image, original_image):
        """
        Detect and extract tables from image
        """
        tables = []
        
        try:
            # Convert PIL to OpenCV
            opencv_image = cv.cvtColor(np.array(processed_image), cv.COLOR_GRAY2BGR)
            gray = cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
            vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv.morphologyEx(gray, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv.morphologyEx(gray, cv.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines
            table_mask = cv.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            table_mask = cv.dilate(table_mask, np.ones((3, 3), np.uint8), iterations=2)
            
            # Find contours
            contours, _ = cv.findContours(table_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv.contourArea(contour)
                if area > 5000:  # Minimum table area
                    x, y, w, h = cv.boundingRect(contour)
                    
                    # Extract table region and run OCR
                    table_region = original_image.crop((x, y, x+w, y+h))
                    table_text = pytesseract.image_to_string(table_region, config=r'--oem 3 --psm 6')
                    
                    if len(table_text.strip()) > 20:
                        # Simple table structure detection
                        lines = table_text.strip().split('\n')
                        rows = [line.strip() for line in lines if line.strip()]
                        
                        table_data = {
                            'rows': len(rows),
                            'columns': max([len(row.split()) for row in rows]) if rows else 0,
                            'raw_text': table_text,
                            'structured_data': rows
                        }
                        
                        tables.append({
                            'content': table_data,
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.8  # Fixed confidence for table detection
                        })
            
        except Exception as e:
            print(f"     Warning: Table detection failed: {e}")
        
        return tables
    
    def _normalize_bbox(self, bbox, image_size):
        """
        Normalize bounding box coordinates to 0-1 range
        """
        width, height = image_size
        x1, y1, x2, y2 = bbox
        return [x1/width, y1/height, x2/width, y2/height]
