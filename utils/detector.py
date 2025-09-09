"""
Page Type Detector
Determines whether a PDF page is text-based or image-based
"""

import fitz  # PyMuPDF


class PageTypeDetector:
    """
    Detects whether a PDF page should be processed as text-based or image-based
    """
    
    def __init__(self, text_threshold=50):
        """
        Initialize the detector
        
        Args:
            text_threshold (int): Minimum characters needed to consider a page text-based
        """
        self.text_threshold = text_threshold
    
    def detect_page_type(self, pdf_path, page_number):
        """
        Detect if a page is text-based or image-based
        
        Args:
            pdf_path (str): Path to PDF file
            page_number (int): Page number (1-based)
            
        Returns:
            str: 'text' or 'image' based on detection
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]  # Convert to 0-based
            
            # Method 1: Try to extract text directly
            text_content = page.get_text()
            text_length = len(text_content.strip())
            
            # Method 2: Check for embedded text objects
            text_dict = page.get_text("dict")
            has_text_blocks = len(text_dict.get("blocks", [])) > 0
            
            # Method 3: Check if page contains only images
            page_area = page.rect.width * page.rect.height
            image_coverage = 0
            
            try:
                images = page.get_images()
                for img_index, img in enumerate(images):
                    try:
                        # Get image bbox to calculate coverage
                        img_dict = page.get_image_info(img_index)
                        if img_dict:
                            img_area = (img_dict.get('width', 0) * img_dict.get('height', 0))
                            image_coverage += img_area
                    except:
                        continue
            except:
                pass
            
            doc.close()
            
            # Decision logic
            print(f"   Page {page_number} analysis:")
            print(f"   - Text length: {text_length}")
            print(f"   - Has text blocks: {has_text_blocks}")
            print(f"   - Image coverage ratio: {image_coverage/page_area if page_area > 0 else 0:.2f}")
            
            # If sufficient extractable text, consider it text-based
            if text_length >= self.text_threshold and has_text_blocks:
                return 'text'
            else:
                return 'image'
                
        except Exception as e:
            print(f"   Error detecting page type for page {page_number}: {e}")
            # Default to image-based if detection fails
            return 'image'
    
    def analyze_pdf_composition(self, pdf_path):
        """
        Analyze the entire PDF to understand its composition
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: Analysis results with page types and statistics
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            page_types = {}
            text_pages = 0
            image_pages = 0
            
            for page_num in range(1, total_pages + 1):
                page_type = self.detect_page_type(pdf_path, page_num)
                page_types[page_num] = page_type
                
                if page_type == 'text':
                    text_pages += 1
                else:
                    image_pages += 1
            
            return {
                'total_pages': total_pages,
                'text_based_pages': text_pages,
                'image_based_pages': image_pages,
                'page_types': page_types,
                'composition_ratio': {
                    'text_percentage': (text_pages / total_pages * 100) if total_pages > 0 else 0,
                    'image_percentage': (image_pages / total_pages * 100) if total_pages > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"Error analyzing PDF composition: {e}")
            return None
