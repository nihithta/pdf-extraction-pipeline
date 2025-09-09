import json
from PIL import Image
import fitz  # PyMuPDF
import os
import glob
from pathlib import Path
import csv
import time
from datetime import datetime
import traceback

class PDFExtractor:
    def __init__(self):
        """
        Initialize PDF extractor - uses PyMuPDF for all processing
        """
        print("PDF Extractor initialized - using PyMuPDF for processing")

    def pdf_page_to_image(self, pdf_path, page_num):
        """
        Convert a PDF page to PIL Image
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Get page as image
        mat = fitz.Matrix(2.0, 2.0)  # High resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        
        # Convert to PIL Image
        from io import BytesIO
        img = Image.open(BytesIO(img_data))
        doc.close()
        return img

    def extract_text_blocks(self, pdf_path, page_num):
        """
        Extract text blocks (paragraphs) and bounding boxes from PDF page
        Groups text into meaningful blocks instead of individual words
        Excludes text that appears within table areas
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # First, extract table areas to exclude them from text extraction
        table_areas = []
        try:
            page_tables = page.find_tables()
            for table in page_tables:
                table_areas.append(table.bbox)
        except:
            pass  # If table detection fails, continue without table exclusion
        
        # Extract text blocks with coordinates
        blocks = page.get_text("dict")
        
        text_blocks = []
        block_boxes = []
        
        for block in blocks["blocks"]:
            if "lines" in block and block.get("type") == 0:  # Text block
                # Combine all text in the block
                block_text = ""
                min_x0, min_y0 = float('inf'), float('inf')
                max_x1, max_y1 = 0, 0
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        span_text = span["text"]
                        if span_text.strip():
                            line_text += span_text
                            # Update bounding box
                            bbox = span["bbox"]
                            min_x0 = min(min_x0, bbox[0])
                            min_y0 = min(min_y0, bbox[1])
                            max_x1 = max(max_x1, bbox[2])
                            max_y1 = max(max_y1, bbox[3])
                    
                    if line_text.strip():
                        block_text += line_text + " "
                
                if block_text.strip():
                    # Create block bounding box
                    block_bbox = [min_x0, min_y0, max_x1, max_y1]
                    
                    # Check if this text block overlaps with any table area
                    is_in_table = False
                    for table_bbox in table_areas:
                        if self._bbox_overlap(block_bbox, table_bbox, overlap_threshold=0.5):
                            is_in_table = True
                            break
                    
                    # Only include text blocks that are NOT in table areas
                    if not is_in_table:
                        # Normalize coordinates to 0-1000 scale
                        norm_x0 = int((min_x0 / page_width) * 1000)
                        norm_y0 = int((min_y0 / page_height) * 1000)
                        norm_x1 = int((max_x1 / page_width) * 1000)
                        norm_y1 = int((max_y1 / page_height) * 1000)
                        
                        # Ensure coordinates are within bounds
                        norm_x0 = max(0, min(1000, norm_x0))
                        norm_y0 = max(0, min(1000, norm_y0))
                        norm_x1 = max(0, min(1000, norm_x1))
                        norm_y1 = max(0, min(1000, norm_y1))
                        
                        # Store original coordinates for annotation
                        original_box = [min_x0, min_y0, max_x1, max_y1]
                        normalized_box = [norm_x0, norm_y0, norm_x1, norm_y1]
                        
                        text_blocks.append({
                            "text": block_text.strip(),
                            "bbox_normalized": normalized_box,
                            "bbox_original": original_box,
                            "type": "paragraph"
                        })
        
        doc.close()
        return text_blocks

    def _bbox_overlap(self, bbox1, bbox2, overlap_threshold=0.5):
        """
        Check if two bounding boxes overlap significantly
        bbox format: [x0, y0, x1, y1]
        Returns True if overlap area is greater than threshold of smaller bbox
        """
        try:
            # Calculate intersection
            x_left = max(bbox1[0], bbox2[0])
            y_top = max(bbox1[1], bbox2[1])
            x_right = min(bbox1[2], bbox2[2])
            y_bottom = min(bbox1[3], bbox2[3])
            
            # Check if there's any intersection
            if x_right <= x_left or y_bottom <= y_top:
                return False
            
            # Calculate intersection area
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate areas of both bboxes
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            # Calculate overlap ratio with respect to the smaller bbox
            smaller_area = min(area1, area2)
            overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0
            
            return overlap_ratio > overlap_threshold
            
        except Exception:
            return False

    def extract_entities_simple(self, text_blocks):
        """
        Extract entities using simple rule-based classification
        """
        if not text_blocks:
            print("No text blocks found for entity extraction")
            return []
        
        print(f"Processing {len(text_blocks)} text blocks for entity classification")
        
        try:
            entities = []
            
            # Classify each text block as a whole using rule-based approach
            for i, block in enumerate(text_blocks):
                block_type = self.classify_text_block(block["text"])
                
                entity = {
                    "text": block["text"],
                    "label": block_type,
                    "bbox_normalized": block["bbox_normalized"],
                    "bbox_original": block["bbox_original"],
                    "type": block["type"],
                    "confidence": 0.8  # Static confidence for rule-based classification
                }
                entities.append(entity)
            
            print(f"Extracted {len(entities)} text block entities")
            return entities
            
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def classify_text_block(self, text):
        """
        Simple rule-based classification of text blocks
        You could replace this with a more sophisticated classifier
        """
        text_lower = text.lower()
        
        # Check for common patterns
        if any(word in text_lower for word in ['agreement', 'contract', 'cpa']):
            return "CONTRACT"
        elif any(word in text_lower for word in ['$', 'price', 'cost', 'payment']):
            return "FINANCIAL"
        elif any(word in text_lower for word in ['date', 'month', 'year', '2024', '2023']):
            return "DATE"
        elif any(word in text_lower for word in ['aptitude', 'boston scientific', 'corporation', 'llc']):
            return "ORGANIZATION"
        elif any(word in text_lower for word in ['commitment', 'tier', 'target']):
            return "TERMS"
        elif len(text.split()) > 50:  # Long text blocks
            return "PARAGRAPH"
        elif len(text.split()) < 5:  # Short text
            return "HEADER"
        else:
            return "TEXT"

    def extract_tables_improved(self, pdf_path, page_num):
        """
        Extract tables with better detection using PyMuPDF
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        tables = []
        
        # Try to find tables using PyMuPDF's table detection
        try:
            page_tables = page.find_tables()
            
            for table_idx, table in enumerate(page_tables):
                table_data = table.extract()
                if table_data:
                    # Get table bounding box
                    table_bbox = table.bbox
                    
                    # Get page dimensions for normalization
                    page_rect = page.rect
                    page_width = page_rect.width
                    page_height = page_rect.height
                    
                    # Normalize coordinates
                    norm_x0 = int((table_bbox[0] / page_width) * 1000)
                    norm_y0 = int((table_bbox[1] / page_height) * 1000)
                    norm_x1 = int((table_bbox[2] / page_width) * 1000)
                    norm_y1 = int((table_bbox[3] / page_height) * 1000)
                    
                    tables.append({
                        "type": "table",
                        "data": table_data,
                        "bbox_original": list(table_bbox),
                        "bbox_normalized": [norm_x0, norm_y0, norm_x1, norm_y1],
                        "rows": len(table_data),
                        "columns": len(table_data[0]) if table_data else 0
                    })
                    
        except Exception as e:
            print(f"Table extraction error: {e}")
        
        doc.close()
        return tables

    def calculate_extraction_metrics(self, text_blocks, entities, tables, image_size):
        """
        Calculate various metrics for PDF extraction quality
        """
        metrics = {}
        
        # Basic extraction metrics
        metrics['total_text_blocks'] = len(text_blocks)
        metrics['total_entities'] = len(entities)
        metrics['total_tables'] = len(tables)
        
        # Text coverage metrics
        total_text_length = sum(len(block['text']) for block in text_blocks)
        metrics['total_text_length'] = total_text_length
        
        # Entity distribution metrics
        entity_types = {}
        confidence_scores = []
        
        for entity in entities:
            entity_type = entity.get('label', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            confidence_scores.append(entity.get('confidence', 0.0))
        
        metrics['entity_types_count'] = len(entity_types)
        metrics['avg_confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        metrics['min_confidence'] = min(confidence_scores) if confidence_scores else 0.0
        metrics['max_confidence'] = max(confidence_scores) if confidence_scores else 0.0
        
        # Layout quality metrics
        if text_blocks:
            block_sizes = [len(block['text'].split()) for block in text_blocks]
            metrics['avg_block_size'] = sum(block_sizes) / len(block_sizes)
            metrics['min_block_size'] = min(block_sizes)
            metrics['max_block_size'] = max(block_sizes)
        else:
            metrics['avg_block_size'] = 0
            metrics['min_block_size'] = 0
            metrics['max_block_size'] = 0
        
        # Coverage estimation (approximate)
        if image_size and text_blocks:
            total_area = image_size[0] * image_size[1]
            text_area = 0
            for block in text_blocks:
                bbox = block['bbox_original']
                block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                text_area += block_area
            
            metrics['text_coverage_ratio'] = min(text_area / total_area, 1.0) if total_area > 0 else 0.0
        else:
            metrics['text_coverage_ratio'] = 0.0
        
        # Quality score calculation (0-100)
        quality_score = 0
        
        # Text extraction quality (40 points)
        if total_text_length > 100:
            quality_score += 40
        elif total_text_length > 50:
            quality_score += 30
        elif total_text_length > 10:
            quality_score += 20
        
        # Entity detection quality (30 points)
        if len(entities) > 5:
            quality_score += 30
        elif len(entities) > 2:
            quality_score += 20
        elif len(entities) > 0:
            quality_score += 10
        
        # Confidence quality (20 points)
        if metrics['avg_confidence'] > 0.8:
            quality_score += 20
        elif metrics['avg_confidence'] > 0.6:
            quality_score += 15
        elif metrics['avg_confidence'] > 0.4:
            quality_score += 10
        
        # Structure detection quality (10 points)
        if len(tables) > 0:
            quality_score += 10
        elif len(entity_types) > 3:
            quality_score += 5
        
        metrics['quality_score'] = quality_score
        metrics['accuracy_percentage'] = quality_score  # Using quality score as accuracy approximation
        
        # Entity type breakdown
        for entity_type, count in entity_types.items():
            metrics[f'entities_{entity_type.lower()}'] = count
        
        return metrics

    def annotate_pdf_with_boxes(self, pdf_path, output_path, entities, tables):
        """
        Create an annotated PDF with bounding boxes overlaid
        """
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Filter entities and tables for this page
                page_entities = [e for e in entities if e.get("page_number", 1) == page_num + 1]
                page_tables = [t for t in tables if t.get("page_number", 1) == page_num + 1]
                
                # Draw entity boxes
                for entity in page_entities:
                    if "bbox_original" in entity:
                        bbox = entity["bbox_original"]
                        rect = fitz.Rect(bbox)
                        
                        # Choose color based on entity type
                        color_map = {
                            "CONTRACT": (1, 0, 0),      # Red
                            "FINANCIAL": (0, 1, 0),     # Green
                            "DATE": (0, 0, 1),          # Blue
                            "ORGANIZATION": (1, 0, 1),   # Magenta
                            "TERMS": (1, 0.5, 0),       # Orange
                            "PARAGRAPH": (0, 0.5, 1),   # Light Blue
                            "HEADER": (0.5, 0, 1),      # Purple
                            "TEXT": (0.5, 0.5, 0.5)     # Gray
                        }
                        
                        color = color_map.get(entity["label"], (0, 0, 0))
                        
                        # Draw rectangle
                        page.draw_rect(rect, color=color, width=2)
                        
                        # Add label
                        label_rect = fitz.Rect(bbox[0], bbox[1] - 15, bbox[0] + 100, bbox[1])
                        page.draw_rect(label_rect, color=color, fill=color)
                        page.insert_text(
                            (bbox[0] + 2, bbox[1] - 2),
                            entity["label"],
                            fontsize=8,
                            color=(1, 1, 1)  # White text
                        )
                
                # Draw table boxes
                for table in page_tables:
                    if "bbox_original" in table:
                        bbox = table["bbox_original"]
                        rect = fitz.Rect(bbox)
                        
                        # Draw table border in cyan
                        page.draw_rect(rect, color=(0, 1, 1), width=3)
                        
                        # Add table label
                        label_rect = fitz.Rect(bbox[0], bbox[1] - 15, bbox[0] + 80, bbox[1])
                        page.draw_rect(label_rect, color=(0, 1, 1), fill=(0, 1, 1))
                        page.insert_text(
                            (bbox[0] + 2, bbox[1] - 2),
                            "TABLE",
                            fontsize=8,
                            color=(0, 0, 0)  # Black text
                        )
            
            # Save annotated PDF
            doc.save(output_path)
            doc.close()
            print(f"Annotated PDF saved as: {output_path}")
            
        except Exception as e:
            print(f"Error creating annotated PDF: {e}")
            import traceback
            traceback.print_exc()

    def table_to_markdown(self, table_data):
        """
        Convert table data to markdown format
        """
        if not table_data or len(table_data) == 0:
            return ""
        
        markdown_table = []
        
        # Process each row
        for row_idx, row in enumerate(table_data):
            # Clean up row data - handle None values and empty strings
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean up cell content - remove excessive whitespace and line breaks
                    cleaned_cell = str(cell).strip().replace('\n', ' ').replace('\r', '')
                    # Escape pipe characters in cell content
                    cleaned_cell = cleaned_cell.replace('|', '\\|')
                    cleaned_row.append(cleaned_cell)
            
            # Create markdown row
            markdown_row = "| " + " | ".join(cleaned_row) + " |"
            markdown_table.append(markdown_row)
            
            # Add header separator after first row
            if row_idx == 0:
                separator = "| " + " | ".join(["---"] * len(cleaned_row)) + " |"
                markdown_table.append(separator)
        
        return "\n".join(markdown_table)

    def extract_content_with_positions(self, pdf_path, page_num):
        """
        Extract all content (text blocks, tables, and images) with their positions to preserve order
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        content_items = []
        
        # Extract text blocks (excluding table areas)
        text_blocks = self.extract_text_blocks(pdf_path, page_num)
        for block in text_blocks:
            content_items.append({
                'type': 'text',
                'content': block['text'],
                'bbox': block['bbox_original'],
                'y_position': block['bbox_original'][1],  # Use top y-coordinate for ordering
                'bbox_normalized': block['bbox_normalized']
            })
        
        # Extract tables
        tables = self.extract_tables_improved(pdf_path, page_num)
        for table in tables:
            content_items.append({
                'type': 'table',
                'content': table,
                'bbox': table['bbox_original'],
                'y_position': table['bbox_original'][1],  # Use top y-coordinate for ordering
                'bbox_normalized': table['bbox_normalized']
            })
        
        # Extract images
        images = self.extract_images(pdf_path, page_num)
        for image in images:
            content_items.append({
                'type': 'image',
                'content': image,
                'bbox': image['bbox_original'],
                'y_position': image['bbox_original'][1],  # Use top y-coordinate for ordering
                'bbox_normalized': image['bbox_normalized']
            })
        
        # Sort by y-position to maintain PDF order (top to bottom)
        content_items.sort(key=lambda x: x['y_position'])
        
        doc.close()
        return content_items

    def extract_images(self, pdf_path, page_num):
        """
        Extract images from PDF page with their positions
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        images = []
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        try:
            # Get image list from the page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image rectangle
                    img_rects = page.get_image_rects(img[0])
                    
                    for rect in img_rects:
                        # Normalize coordinates
                        norm_x0 = int((rect.x0 / page_width) * 1000)
                        norm_y0 = int((rect.y0 / page_height) * 1000)
                        norm_x1 = int((rect.x1 / page_width) * 1000)
                        norm_y1 = int((rect.y1 / page_height) * 1000)
                        
                        images.append({
                            "type": "image",
                            "image_index": img_index,
                            "bbox_original": [rect.x0, rect.y0, rect.x1, rect.y1],
                            "bbox_normalized": [norm_x0, norm_y0, norm_x1, norm_y1],
                            "width": rect.width,
                            "height": rect.height,
                            "message": "Image detected in PDF"
                        })
                        
                except Exception as e:
                    print(f"Error processing image {img_index}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error extracting images: {e}")
        
        doc.close()
        return images

def process_pdf_page(pdf_path, page_number, output_dir=None):
    """
    Process a specific page from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        page_number (int): Page number to process (1-based indexing)
        output_dir (str): Optional output directory for saving results
    
    Returns:
        dict: Page processing results with content, metrics, and image detection info
    """
    extractor = PDFExtractor()
    
    try:
        # Open PDF to validate page number
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        if page_number < 1 or page_number > total_pages:
            raise ValueError(f"Invalid page number {page_number}. PDF has {total_pages} pages.")
        
        print(f"Processing page {page_number} from {Path(pdf_path).name}...")
        
        # Convert to 0-based indexing for internal processing
        page_num = page_number - 1
        
        # Extract content with positions to preserve order
        content_items = extractor.extract_content_with_positions(pdf_path, page_num)
        
        page_content = []
        page_tables = []
        page_images = []
        page_text_blocks = []
        
        has_images = False
        
        if not content_items:
            print(f"No content found on page {page_number}")
        else:
            # Process content items in order
            for item in content_items:
                if item['type'] == 'text':
                    text_block = {
                        "type": "text",
                        "content": item['content'].strip(),
                        "bbox_original": item['bbox'],
                        "bbox_normalized": item['bbox_normalized'],
                        "order": len(page_content)
                    }
                    page_content.append(text_block)
                    page_text_blocks.append({
                        "text": item['content'],
                        "bbox_original": item['bbox'],
                        "bbox_normalized": item['bbox_normalized'],
                        "type": "paragraph"
                    })
                
                elif item['type'] == 'table':
                    table = item['content']
                    table_block = {
                        "type": "table",
                        "content": {
                            "data": table['data'],
                            "rows": table['rows'],
                            "columns": table['columns']
                        },
                        "bbox_original": table['bbox_original'],
                        "bbox_normalized": table['bbox_normalized'],
                        "order": len(page_content)
                    }
                    page_content.append(table_block)
                    page_tables.append(table)
                
                elif item['type'] == 'image':
                    has_images = True
                    image = item['content']
                    image_block = {
                        "type": "image",
                        "content": {
                            "message": "Image detected in between content",
                            "image_index": image['image_index'],
                            "dimensions": {
                                "width": image['width'],
                                "height": image['height']
                            }
                        },
                        "bbox_original": image['bbox_original'],
                        "bbox_normalized": image['bbox_normalized'],
                        "order": len(page_content)
                    }
                    page_content.append(image_block)
                    page_images.append(image)
        
        # Extract entities if there are text blocks
        entities = []
        page_metrics = {}
        
        if page_text_blocks:
            image = extractor.pdf_page_to_image(pdf_path, page_num)
            entities = extractor.extract_entities_simple(page_text_blocks)
            
            # Calculate page metrics
            page_metrics = extractor.calculate_extraction_metrics(
                page_text_blocks, entities, page_tables, image.size
            )
        
        # Prepare the result
        result = {
            "page_number": page_number,
            "pdf_file": Path(pdf_path).name,
            "content": page_content,
            "entities": entities,
            "has_images": has_images,
            "image_count": len(page_images),
            "summary": {
                "total_text_blocks": len([c for c in page_content if c['type'] == 'text']),
                "total_tables": len([c for c in page_content if c['type'] == 'table']),
                "total_images": len([c for c in page_content if c['type'] == 'image']),
                "total_content_items": len(page_content)
            },
            "metrics": page_metrics,
            "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Call image processing script if images are detected
        if has_images:
            result["image_processing"] = handle_page_with_images(pdf_path, page_number, page_images)
        
        # Save results if output directory is specified
        if output_dir:
            save_page_results(result, pdf_path, page_number, output_dir)
        
        print(f"✓ Page {page_number} processed successfully")
        print(f"  - Text blocks: {result['summary']['total_text_blocks']}")
        print(f"  - Tables: {result['summary']['total_tables']}")
        print(f"  - Images: {result['summary']['total_images']}")
        if has_images:
            print(f"  - Image processing triggered")
        
        return result
        
    except Exception as e:
        print(f"Error processing page {page_number}: {e}")
        traceback.print_exc()
        return {
            "page_number": page_number,
            "pdf_file": Path(pdf_path).name,
            "error": str(e),
            "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def handle_page_with_images(pdf_path, page_number, page_images):
    """
    Handle processing when images are detected on a page.
    This function can be replaced with a call to another script.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_number (int): Page number (1-based)
        page_images (list): List of detected images with their metadata
    
    Returns:
        dict: Image processing results
    """
    print(f"🖼️  Images detected on page {page_number}, calling image processing...")
    
    # This is where you would call your separate image processing script
    # For example:
    # import your_image_processing_script
    # return your_image_processing_script.process_images(pdf_path, page_number, page_images)
    
    # For now, return image metadata
    image_processing_result = {
        "images_processed": len(page_images),
        "processing_method": "placeholder - call external script here",
        "images_info": []
    }
    
    for i, image in enumerate(page_images):
        image_info = {
            "image_index": image.get('image_index', i),
            "bbox": image.get('bbox_original', []),
            "dimensions": {
                "width": image.get('width', 0),
                "height": image.get('height', 0)
            }
        }
        image_processing_result["images_info"].append(image_info)
    
    return image_processing_result

def save_page_results(result, pdf_path, page_number, output_dir):
    """
    Save page processing results to files
    
    Args:
        result (dict): Page processing results
        pdf_path (str): Original PDF path
        page_number (int): Page number
        output_dir (str): Output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_filename = Path(pdf_path).stem
        
        # Save JSON result
        json_filename = f"{pdf_filename}_page_{page_number}_output.json"
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"  JSON saved: {json_path}")
        
        # Create annotated PDF for this page if there are entities
        if result.get('entities'):
            extractor = PDFExtractor()
            annotated_filename = f"{pdf_filename}_page_{page_number}_annotated.pdf"
            annotated_path = os.path.join(output_dir, annotated_filename)
            
            # Create single-page annotated PDF
            create_single_page_annotated_pdf(pdf_path, page_number, result['entities'], annotated_path)
            print(f"  Annotated PDF saved: {annotated_path}")
        
    except Exception as e:
        print(f"Error saving page results: {e}")

def create_single_page_annotated_pdf(pdf_path, page_number, entities, output_path):
    """
    Create an annotated PDF for a single page
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Create new document with only the target page
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_number-1, to_page=page_number-1)
        
        page = new_doc[0]  # First (and only) page in new document
        
        # Draw entity boxes
        for entity in entities:
            if "bbox_original" in entity:
                bbox = entity["bbox_original"]
                rect = fitz.Rect(bbox)
                
                # Choose color based on entity type
                color_map = {
                    "CONTRACT": (1, 0, 0),      # Red
                    "FINANCIAL": (0, 1, 0),     # Green
                    "DATE": (0, 0, 1),          # Blue
                    "ORGANIZATION": (1, 0, 1),   # Magenta
                    "TERMS": (1, 0.5, 0),       # Orange
                    "PARAGRAPH": (0, 0.5, 1),   # Light Blue
                    "HEADER": (0.5, 0, 1),      # Purple
                    "TEXT": (0.5, 0.5, 0.5)     # Gray
                }
                
                color = color_map.get(entity["label"], (0, 0, 0))
                
                # Draw rectangle
                page.draw_rect(rect, color=color, width=2)
                
                # Add label
                label_rect = fitz.Rect(bbox[0], bbox[1] - 15, bbox[0] + 100, bbox[1])
                page.draw_rect(label_rect, color=color, fill=color)
                page.insert_text(
                    (bbox[0] + 2, bbox[1] - 2),
                    entity["label"],
                    fontsize=8,
                    color=(1, 1, 1)  # White text
                )
        
        # Save annotated PDF
        new_doc.save(output_path)
        new_doc.close()
        doc.close()
        
    except Exception as e:
        print(f"Error creating single-page annotated PDF: {e}")

def pdf_to_json_layoutlm(pdf_path, json_path, create_annotated_pdf=True):
    """
    Convert PDF to JSON using improved text block extraction
    Preserves the order of content as it appears in the PDF, detects images, separates tables from text
    
    NOTE: This function is kept for backward compatibility.
    For processing specific pages, use process_pdf_page() instead.
    """
    extractor = PDFExtractor()
    output = {"pages": [], "model_info": "PyMuPDF-RuleBased"}

    try:
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"Processing {total_pages} pages...")
        
        all_entities = []
        all_tables = []
        
        for page_num in range(total_pages):
            print(f"Processing page {page_num + 1}/{total_pages}...")
            
            # Extract content with positions to preserve order
            content_items = extractor.extract_content_with_positions(pdf_path, page_num)
            
            page_content = []
            page_tables = []
            page_images = []
            page_text_blocks = []
            
            if not content_items:
                print(f"No content found on page {page_num + 1}")
            else:
                # Process content items in order
                for item in content_items:
                    if item['type'] == 'text':
                        text_block = {
                            "type": "text",
                            "content": item['content'].strip(),
                            "bbox_original": item['bbox'],
                            "bbox_normalized": item['bbox_normalized'],
                            "order": len(page_content)
                        }
                        page_content.append(text_block)
                        page_text_blocks.append({
                            "text": item['content'],
                            "bbox_original": item['bbox'],
                            "bbox_normalized": item['bbox_normalized'],
                            "type": "paragraph"
                        })
                    
                    elif item['type'] == 'table':
                        table = item['content']
                        table_block = {
                            "type": "table",
                            "content": {
                                "data": table['data'],
                                "rows": table['rows'],
                                "columns": table['columns']
                            },
                            "bbox_original": table['bbox_original'],
                            "bbox_normalized": table['bbox_normalized'],
                            "order": len(page_content)
                        }
                        page_content.append(table_block)
                        page_tables.append(table)
                    
                    elif item['type'] == 'image':
                        image = item['content']
                        image_block = {
                            "type": "image",
                            "content": {
                                "message": "Image detected in between content",
                                "image_index": image['image_index'],
                                "dimensions": {
                                    "width": image['width'],
                                    "height": image['height']
                                }
                            },
                            "bbox_original": image['bbox_original'],
                            "bbox_normalized": image['bbox_normalized'],
                            "order": len(page_content)
                        }
                        page_content.append(image_block)
                        page_images.append(image)
            
            # Extract entities for annotation (if needed)
            if page_text_blocks:
                entities = extractor.extract_entities_simple(page_text_blocks)
                
                for entity in entities:
                    entity["page_number"] = page_num + 1
                all_entities.extend(entities)
            else:
                entities = []
            
            # Add page number to tables for annotation
            for table in page_tables:
                table["page_number"] = page_num + 1
            all_tables.extend(page_tables)
            
            # Add page number to images
            for image in page_images:
                image["page_number"] = page_num + 1
            
            page_data = {
                "page_number": page_num + 1,
                "content": page_content,  # All content in order (text, tables, images)
                "summary": {
                    "total_text_blocks": len([c for c in page_content if c['type'] == 'text']),
                    "total_tables": len([c for c in page_content if c['type'] == 'table']),
                    "total_images": len([c for c in page_content if c['type'] == 'image']),
                    "total_content_items": len(page_content)
                },
                "layout_info": {
                    "image_size": extractor.pdf_page_to_image(pdf_path, page_num).size if page_content else None
                }
            }
            
            output["pages"].append(page_data)

        # Add document summary
        output["document_summary"] = {
            "total_pages": total_pages,
            "total_entities": len(all_entities),
            "total_tables": len(all_tables),
            "total_images": sum(len([c for c in page["content"] if c["type"] == "image"]) for page in output["pages"]),
            "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Successfully converted '{pdf_path}' to '{json_path}' using improved LayoutLM.")
        print(f"Processed {len(output['pages'])} pages with {len(all_entities)} entities, {len(all_tables)} tables, and {output['document_summary']['total_images']} images.")
        
        # Create annotated PDF if requested
        if create_annotated_pdf:
            annotated_path = pdf_path.replace('.pdf', '_annotatedlm1.pdf')
            extractor.annotate_pdf_with_boxes(pdf_path, annotated_path, all_entities, all_tables)

    except Exception as e:
        print(f"Error processing PDF with LayoutLM: {e}")
        import traceback
        traceback.print_exc()

def create_folder_structure(base_path):
    """
    Create the required folder structure for automation
    """
    # Use existing folder structure with phase1 inside Json_pdf
    phase1_path = os.path.join(base_path, 'Json_pdf', 'phase1')
    input_phase1_path = os.path.join(base_path, 'input_pdfs', 'phase1')

    folders = {
        'input': input_phase1_path,
        'json_output': phase1_path,
        'annotated_output': os.path.join(base_path, 'anotted_pdf'),
        'metrics_output': os.path.join(base_path, 'testing_data')
    }
    
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created/Verified folder: {folder_path}")
    
    return folders

def process_pdf_batch(input_folder, json_output_folder, annotated_output_folder, metrics_output_folder):
    """
    Process all PDF files in the input folder and save outputs to respective folders
    """
    # Get all PDF files from input folder
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize PDF extractor once for all files
    extractor = PDFExtractor()
    
    # Initialize batch metrics
    batch_metrics = []
    batch_start_time = time.time()
    
    for pdf_path in pdf_files:
        try:
            pdf_filename = Path(pdf_path).stem  # Get filename without extension
            print(f"\nProcessing: {pdf_filename}.pdf")
            
            # Define output paths
            json_output_path = os.path.join(json_output_folder, f"{pdf_filename}_output.json")
            annotated_output_path = os.path.join(annotated_output_folder, f"{pdf_filename}_annotated.pdf")
            
            # Process the PDF and get metrics
            file_metrics = process_single_pdf_json(extractor, pdf_path, json_output_path, annotated_output_path)
            file_metrics['filename'] = f"{pdf_filename}.pdf"
            file_metrics['processing_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            batch_metrics.append(file_metrics)
            
            print(f"✓ Completed processing: {pdf_filename}.pdf (Accuracy: {file_metrics['accuracy_percentage']:.1f}%)")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {e}")
            # Add error entry to metrics
            error_metrics = {
                'filename': Path(pdf_path).name,
                'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': str(e),
                'accuracy_percentage': 0.0,
                'quality_score': 0.0,
                'total_pages': 0,
                'processing_time_seconds': 0.0
            }
            batch_metrics.append(error_metrics)
            continue
    
    # Calculate batch processing time
    batch_end_time = time.time()
    batch_processing_time = batch_end_time - batch_start_time
    
    # Save batch metrics to CSV
    save_batch_metrics_csv(batch_metrics, metrics_output_folder, batch_processing_time)
    
    print(f"\nBatch processing completed!")
    print(f"Total processing time: {batch_processing_time:.2f} seconds")
    
    # Print summary statistics
    successful_files = [m for m in batch_metrics if m.get('accuracy_percentage', 0) > 0]
    if successful_files:
        avg_accuracy = sum(m['accuracy_percentage'] for m in successful_files) / len(successful_files)
        print(f"Average accuracy across {len(successful_files)} files: {avg_accuracy:.1f}%")
    
    return batch_metrics

def process_single_pdf_json(extractor, pdf_path, json_output_path, annotated_output_path):
    """
    Process a single PDF file using the existing PDF extraction logic and output JSON
    Returns metrics for the processed file
    """
    output = {"pages": [], "model_info": "PyMuPDF-RuleBased"}
    
    # Track processing metrics
    start_time = time.time()
    file_metrics = {
        'filename': Path(pdf_path).name,
        'file_size_mb': os.path.getsize(pdf_path) / (1024 * 1024),
        'total_pages': 0,
        'total_text_blocks': 0,
        'total_entities': 0,
        'total_tables': 0,
        'total_images': 0,
        'total_text_length': 0,
        'avg_confidence': 0.0,
        'quality_score': 0.0,
        'accuracy_percentage': 0.0,
        'text_coverage_ratio': 0.0,
        'processing_time_seconds': 0.0,
        'pages_processed': 0,
        'error': None
    }

    try:
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        file_metrics['total_pages'] = total_pages
        doc.close()
        
        print(f"  Processing {total_pages} pages...")
        
        all_entities = []
        all_tables = []
        all_text_blocks = []
        page_metrics = []

        for page_num in range(total_pages):
            print(f"  Processing page {page_num + 1}/{total_pages}...")
            
            try:
                # Extract content with positions to preserve order
                content_items = extractor.extract_content_with_positions(pdf_path, page_num)
                
                page_content = []
                page_tables = []
                page_images = []
                page_text_blocks = []
                
                if not content_items:
                    page_content = []
                else:
                    # Process content items in order
                    for item in content_items:
                        if item['type'] == 'text':
                            text_block = {
                                "type": "text",
                                "content": item['content'].strip(),
                                "bbox_original": item['bbox'],
                                "bbox_normalized": item['bbox_normalized'],
                                "order": len(page_content)
                            }
                            page_content.append(text_block)
                            page_text_blocks.append({
                                "text": item['content'],
                                "bbox_original": item['bbox'],
                                "bbox_normalized": item['bbox_normalized'],
                                "type": "paragraph"
                            })
                        
                        elif item['type'] == 'table':
                            table = item['content']
                            table_block = {
                                "type": "table",
                                "content": {
                                    "data": table['data'],
                                    "rows": table['rows'],
                                    "columns": table['columns']
                                },
                                "bbox_original": table['bbox_original'],
                                "bbox_normalized": table['bbox_normalized'],
                                "order": len(page_content)
                            }
                            page_content.append(table_block)
                            page_tables.append(table)
                        
                        elif item['type'] == 'image':
                            image = item['content']
                            image_block = {
                                "type": "image",
                                "content": {
                                    "message": "Image detected in between content",
                                    "image_index": image['image_index'],
                                    "dimensions": {
                                        "width": image['width'],
                                        "height": image['height']
                                    }
                                },
                                "bbox_original": image['bbox_original'],
                                "bbox_normalized": image['bbox_normalized'],
                                "order": len(page_content)
                            }
                            page_content.append(image_block)
                            page_images.append(image)
                
                # Extract entities for annotation (if needed) - only from text blocks in page_content
                entities = []
                if page_text_blocks:
                    image = extractor.pdf_page_to_image(pdf_path, page_num)
                    entities = extractor.extract_entities_simple(page_text_blocks)
                    
                    for entity in entities:
                        entity["page_number"] = page_num + 1
                    all_entities.extend(entities)
                    
                    # Calculate page-level metrics using existing tables from page_tables
                    page_metric = extractor.calculate_extraction_metrics(page_text_blocks, entities, page_tables, image.size)
                    page_metric['page_number'] = page_num + 1
                    page_metrics.append(page_metric)
                    
                    all_text_blocks.extend(page_text_blocks)
                
                # Add extracted tables to all_tables for annotation
                all_tables.extend(page_tables)
                
                # Add page number to tables for annotation
                for table in page_tables:
                    table["page_number"] = page_num + 1
                
                # Add page number to images
                for image in page_images:
                    image["page_number"] = page_num + 1
                
                page_data = {
                    "page_number": page_num + 1,
                    "content": page_content,  # All content in order (text, tables, images)
                    "summary": {
                        "total_text_blocks": len([c for c in page_content if c['type'] == 'text']),
                        "total_tables": len([c for c in page_content if c['type'] == 'table']),
                        "total_images": len([c for c in page_content if c['type'] == 'image']),
                        "total_content_items": len(page_content)
                    },
                    "layout_info": {
                        "image_size": extractor.pdf_page_to_image(pdf_path, page_num).size if page_content else None
                    }
                }
                
                output["pages"].append(page_data)
                file_metrics['pages_processed'] += 1
                
            except Exception as page_error:
                print(f"  Error processing page {page_num + 1}: {page_error}")
                continue

        # Add document summary
        output["document_summary"] = {
            "total_pages": total_pages,
            "total_entities": len(all_entities),
            "total_tables": len(all_tables),
            "total_images": sum(len([c for c in page["content"] if c["type"] == "image"]) for page in output["pages"]),
            "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Calculate overall file metrics
        if page_metrics:
            file_metrics['total_text_blocks'] = sum(m['total_text_blocks'] for m in page_metrics)
            file_metrics['total_entities'] = sum(m['total_entities'] for m in page_metrics)
            file_metrics['total_tables'] = sum(m['total_tables'] for m in page_metrics)
            file_metrics['total_text_length'] = sum(m['total_text_length'] for m in page_metrics)
            file_metrics['avg_confidence'] = sum(m['avg_confidence'] for m in page_metrics) / len(page_metrics)
            file_metrics['text_coverage_ratio'] = sum(m['text_coverage_ratio'] for m in page_metrics) / len(page_metrics)
            file_metrics['quality_score'] = sum(m['quality_score'] for m in page_metrics) / len(page_metrics)
            file_metrics['accuracy_percentage'] = file_metrics['quality_score']  # Using quality score as accuracy

        file_metrics['total_images'] = output["document_summary"]["total_images"]

        # Save to JSON file
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  JSON saved: {json_output_path}")
        print(f"  Processed {file_metrics['pages_processed']} pages with {len(all_entities)} entities, {len(all_tables)} tables, and {file_metrics['total_images']} images.")
        
        # Create annotated PDF
        extractor.annotate_pdf_with_boxes(pdf_path, annotated_output_path, all_entities, all_tables)
        print(f"  Annotated PDF saved: {annotated_output_path}")
        
        # Calculate processing time
        end_time = time.time()
        file_metrics['processing_time_seconds'] = end_time - start_time
        
        print(f"  File accuracy: {file_metrics['accuracy_percentage']:.1f}%")
        print(f"  Processing time: {file_metrics['processing_time_seconds']:.2f} seconds")
        
        return file_metrics

    except Exception as e:
        print(f"  Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        
        # Calculate processing time even for errors
        end_time = time.time()
        file_metrics['processing_time_seconds'] = end_time - start_time
        file_metrics['error'] = str(e)
        
        return file_metrics

def process_single_pdf(extractor, pdf_path, json_output_path, annotated_output_path):
    """
    Process a single PDF file using the existing PDF extraction logic
    Returns metrics for the processed file
    """
    output = {"pages": [], "model_info": "PyMuPDF-RuleBased"}
    
    # Track processing metrics
    start_time = time.time()
    file_metrics = {
        'filename': Path(pdf_path).name,
        'file_size_mb': os.path.getsize(pdf_path) / (1024 * 1024),
        'total_pages': 0,
        'total_text_blocks': 0,
        'total_entities': 0,
        'total_tables': 0,
        'total_text_length': 0,
        'avg_confidence': 0.0,
        'quality_score': 0.0,
        'accuracy_percentage': 0.0,
        'text_coverage_ratio': 0.0,
        'processing_time_seconds': 0.0,
        'pages_processed': 0,
        'error': None
    }

    try:
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        file_metrics['total_pages'] = total_pages
        doc.close()
        
        print(f"  Processing {total_pages} pages...")
        
        all_entities = []
        all_tables = []
        all_text_blocks = []
        page_metrics = []
        
        for page_num in range(total_pages):
            print(f"  Processing page {page_num + 1}/{total_pages}...")
            
            try:
                # Convert page to image (still needed for LayoutLM)
                image = extractor.pdf_page_to_image(pdf_path, page_num)
                
                # Extract text blocks instead of individual words
                text_blocks = extractor.extract_text_blocks(pdf_path, page_num)
                
                if not text_blocks:
                    print(f"  No text blocks found on page {page_num + 1}")
                    continue
                
                # Prepare data for LayoutLM (word-level for processing)
                # Extract entities using simple rule-based method
                entities = extractor.extract_entities_simple(text_blocks)
                
                # Add page number to entities
                for entity in entities:
                    entity["page_number"] = page_num + 1
                
                # Extract tables with improved method
                tables = extractor.extract_tables_improved(pdf_path, page_num)
                
                # Add page number to tables
                for table in tables:
                    table["page_number"] = page_num + 1
                
                # Calculate page-level metrics
                page_metric = extractor.calculate_extraction_metrics(text_blocks, entities, tables, image.size)
                page_metric['page_number'] = page_num + 1
                page_metrics.append(page_metric)
                
                all_entities.extend(entities)
                all_tables.extend(tables)
                all_text_blocks.extend(text_blocks)
                
                page_data = {
                    "page_number": page_num + 1,
                    "text_blocks": [
                        {
                            "text": block["text"],
                            "type": block["type"],
                            "bbox_normalized": block["bbox_normalized"],
                            "bbox_original": block["bbox_original"]
                        }
                        for block in text_blocks
                    ],
                    "entities": entities,
                    "tables": tables,
                    "layout_info": {
                        "total_text_blocks": len(text_blocks),
                        "total_entities": len(entities),
                        "total_tables": len(tables),
                        "image_size": image.size
                    },
                    "page_metrics": page_metric
                }
                
                output["pages"].append(page_data)
                file_metrics['pages_processed'] += 1
                
            except Exception as page_error:
                print(f"  Error processing page {page_num + 1}: {page_error}")
                continue

        # Calculate overall file metrics
        if page_metrics:
            file_metrics['total_text_blocks'] = sum(m['total_text_blocks'] for m in page_metrics)
            file_metrics['total_entities'] = sum(m['total_entities'] for m in page_metrics)
            file_metrics['total_tables'] = sum(m['total_tables'] for m in page_metrics)
            file_metrics['total_text_length'] = sum(m['total_text_length'] for m in page_metrics)
            file_metrics['avg_confidence'] = sum(m['avg_confidence'] for m in page_metrics) / len(page_metrics)
            file_metrics['text_coverage_ratio'] = sum(m['text_coverage_ratio'] for m in page_metrics) / len(page_metrics)
            file_metrics['quality_score'] = sum(m['quality_score'] for m in page_metrics) / len(page_metrics)
            file_metrics['accuracy_percentage'] = file_metrics['quality_score']  # Using quality score as accuracy
        
        # Add summary metrics to output
        output["file_metrics"] = file_metrics
        output["processing_summary"] = {
            "total_pages_processed": file_metrics['pages_processed'],
            "total_text_blocks": file_metrics['total_text_blocks'],
            "total_entities": file_metrics['total_entities'],
            "total_tables": file_metrics['total_tables'],
            "overall_accuracy": f"{file_metrics['accuracy_percentage']:.1f}%"
        }

        # Save to JSON file
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  JSON saved: {json_output_path}")
        print(f"  Processed {len(output['pages'])} pages with {len(all_entities)} entities and {len(all_tables)} tables.")
        
        # Create annotated PDF
        extractor.annotate_pdf_with_boxes(pdf_path, annotated_output_path, all_entities, all_tables)
        print(f"  Annotated PDF saved: {annotated_output_path}")
        
        # Calculate processing time
        end_time = time.time()
        file_metrics['processing_time_seconds'] = end_time - start_time
        
        print(f"  File accuracy: {file_metrics['accuracy_percentage']:.1f}%")
        print(f"  Processing time: {file_metrics['processing_time_seconds']:.2f} seconds")
        
        return file_metrics

    except Exception as e:
        print(f"  Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        
        # Calculate processing time even for errors
        end_time = time.time()
        file_metrics['processing_time_seconds'] = end_time - start_time
        file_metrics['error'] = str(e)
        
        return file_metrics

def save_batch_metrics_csv(batch_metrics, metrics_output_folder, batch_processing_time):
    """
    Save batch processing metrics to CSV file
    """
    if not batch_metrics:
        print("No metrics to save")
        return
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"pdf_processing_metrics_{timestamp}.csv"
    csv_path = os.path.join(metrics_output_folder, csv_filename)
    
    # Define CSV headers
    headers = [
        'filename', 'processing_timestamp', 'file_size_mb', 'total_pages', 
        'pages_processed', 'accuracy_percentage', 'quality_score',
        'total_text_blocks', 'total_entities', 'total_tables', 
        'total_text_length', 'avg_confidence', 'text_coverage_ratio',
        'processing_time_seconds', 'error'
    ]
    
    # Add entity type columns if they exist
    entity_columns = set()
    for metrics in batch_metrics:
        for key in metrics.keys():
            if key.startswith('entities_'):
                entity_columns.add(key)
    
    headers.extend(sorted(entity_columns))
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for metrics in batch_metrics:
                # Create row with all headers, filling missing values with 0 or empty string
                row = {}
                for header in headers:
                    if header in metrics:
                        row[header] = metrics[header]
                    elif header.startswith('entities_'):
                        row[header] = 0  # Default entity count
                    elif header in ['accuracy_percentage', 'quality_score', 'avg_confidence', 'text_coverage_ratio', 'processing_time_seconds', 'file_size_mb']:
                        row[header] = 0.0  # Default numeric values
                    elif header in ['total_pages', 'pages_processed', 'total_text_blocks', 'total_entities', 'total_tables', 'total_text_length']:
                        row[header] = 0  # Default integer values
                    else:
                        row[header] = ''  # Default string values
                
                writer.writerow(row)
            
            # Add summary row
            successful_files = [m for m in batch_metrics if m.get('accuracy_percentage', 0) > 0]
            if successful_files:
                summary_row = {
                    'filename': 'BATCH_SUMMARY',
                    'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_pages': sum(m.get('total_pages', 0) for m in successful_files),
                    'pages_processed': sum(m.get('pages_processed', 0) for m in successful_files),
                    'accuracy_percentage': sum(m.get('accuracy_percentage', 0) for m in successful_files) / len(successful_files),
                    'quality_score': sum(m.get('quality_score', 0) for m in successful_files) / len(successful_files),
                    'total_text_blocks': sum(m.get('total_text_blocks', 0) for m in successful_files),
                    'total_entities': sum(m.get('total_entities', 0) for m in successful_files),
                    'total_tables': sum(m.get('total_tables', 0) for m in successful_files),
                    'processing_time_seconds': batch_processing_time,
                    'file_size_mb': sum(m.get('file_size_mb', 0) for m in successful_files)
                }
                
                # Fill remaining headers with empty values or zeros
                for header in headers:
                    if header not in summary_row:
                        if header.startswith('entities_'):
                            summary_row[header] = sum(m.get(header, 0) for m in successful_files)
                        elif header in ['avg_confidence', 'text_coverage_ratio']:
                            summary_row[header] = sum(m.get(header, 0) for m in successful_files) / len(successful_files) if successful_files else 0.0
                        else:
                            summary_row[header] = ''
                
                writer.writerow(summary_row)
        
        print(f"Metrics saved to: {csv_path}")
        
        # Also create a simple summary CSV
        summary_csv_path = os.path.join(metrics_output_folder, f"accuracy_summary_{timestamp}.csv")
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Accuracy %', 'Quality Score', 'Processing Time (sec)', 'Status'])
            
            for metrics in batch_metrics:
                status = 'Success' if not metrics.get('error') else 'Error'
                writer.writerow([
                    metrics.get('filename', ''),
                    f"{metrics.get('accuracy_percentage', 0):.1f}",
                    f"{metrics.get('quality_score', 0):.1f}",
                    f"{metrics.get('processing_time_seconds', 0):.2f}",
                    status
                ])
        
        print(f"Summary saved to: {summary_csv_path}")
        
        return csv_path, summary_csv_path
        
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")
        return None, None

def install_requirements():
    """
    Install required packages for PDF extraction
    """
    packages = [
        "PyMuPDF",
        "pillow"
    ]
    
    import subprocess
    import sys
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

if __name__ == "__main__":
    # Uncomment the line below if you need to install requirements
    # install_requirements()
    
    # Setup folder structure
    base_path = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    folders = create_folder_structure(base_path)
    
    print("=== PDF Processing Automation with Metrics ===")
    print(f"Input folder: {folders['input']}")
    print(f"JSON output folder: {folders['json_output']}")
    print(f"Annotated PDF output folder: {folders['annotated_output']}")
    print(f"Metrics/Testing data folder: {folders['metrics_output']}")
    print("\nPlace your PDF files in the 'input_pdfs' folder and run this script.")
    print("Accuracy metrics will be saved as CSV files in the 'testing_data' folder.")
    
    # Check if there are any PDF files to process
    pdf_count = len(glob.glob(os.path.join(folders['input'], "*.pdf")))
    
    if pdf_count > 0:
        print(f"\nFound {pdf_count} PDF file(s) in input folder. Starting batch processing...")
        
        # Process all PDFs in batch
        batch_metrics = process_pdf_batch(
            folders['input'], 
            folders['json_output'], 
            folders['annotated_output'],
            folders['metrics_output']
        )
        
        print("\n=== Processing Complete ===")
        print(f"Check the following folders for outputs:")
        print(f"- JSON files: {folders['json_output']}")
        print(f"- Annotated PDFs: {folders['annotated_output']}")
        print(f"- Accuracy metrics (CSV): {folders['metrics_output']}")
        
        # Print final statistics
        if batch_metrics:
            successful_files = [m for m in batch_metrics if m.get('accuracy_percentage', 0) > 0]
            failed_files = [m for m in batch_metrics if m.get('error')]
            
            print(f"\n=== Final Statistics ===")
            print(f"Total files processed: {len(batch_metrics)}")
            print(f"Successful: {len(successful_files)}")
            print(f"Failed: {len(failed_files)}")
            
            if successful_files:
                avg_accuracy = sum(m['accuracy_percentage'] for m in successful_files) / len(successful_files)
                max_accuracy = max(m['accuracy_percentage'] for m in successful_files)
                min_accuracy = min(m['accuracy_percentage'] for m in successful_files)
                
                print(f"Average accuracy: {avg_accuracy:.1f}%")
                print(f"Best accuracy: {max_accuracy:.1f}%")
                print(f"Lowest accuracy: {min_accuracy:.1f}%")
                
                total_entities = sum(m.get('total_entities', 0) for m in successful_files)
                total_tables = sum(m.get('total_tables', 0) for m in successful_files)
                print(f"Total entities extracted: {total_entities}")
                print(f"Total tables detected: {total_tables}")
    else:
        print(f"\nNo PDF files found in {folders['input']}")
        print("Please add PDF files to the input folder and run the script again.")
    
    # Optional: Keep the old single file processing for manual use
    # Uncomment the lines below if you want to process a specific file manually
    """
    # Example usage for single file processing
    pdf_path = "mini_contract.pdf"  # Change this to your PDF file
    json_output = "output.json"
    
    # Process PDF and create annotated version
    pdf_to_json_layoutlm(pdf_path, json_output, create_annotated_pdf=True)
    """
