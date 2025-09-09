"""
Hybrid PDF Processor
Main orchestrator that automatically detects page types and routes to appropriate processors
"""

import os
import json
import fitz
from pathlib import Path
from datetime import datetime

from ..utils.detector import PageTypeDetector
from ..utils.metrics import MetricsCalculator
from ..processors.text_processor import TextBasedProcessor
from ..processors.image_processor import ImageBasedProcessor


class HybridPDFProcessor:
    """
    Main hybrid PDF processor that automatically detects page types and routes processing
    """
    
    def __init__(self, text_threshold=50, confidence_threshold=70):
        """
        Initialize the hybrid processor
        
        Args:
            text_threshold (int): Minimum characters needed to consider a page text-based
            confidence_threshold (int): Minimum OCR confidence for image-based processing
        """
        self.detector = PageTypeDetector(text_threshold)
        self.text_processor = TextBasedProcessor()
        self.image_processor = ImageBasedProcessor(confidence_threshold)
        
        print("🚀 Hybrid PDF Processor initialized")
        print(f"   Text threshold: {text_threshold} characters")
        print(f"   OCR confidence threshold: {confidence_threshold}%")
    
    def process_pdf(self, pdf_path, output_dir=None):
        """
        Process entire PDF with hybrid approach
        
        Args:
            pdf_path (str): Path to PDF file
            output_dir (str): Optional output directory
            
        Returns:
            dict: Complete processing results
        """
        print(f"\n{'='*80}")
        print(f"🔄 HYBRID PDF PROCESSING: {Path(pdf_path).name}")
        print(f"{'='*80}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"📄 Total pages: {total_pages}")
        
        # Results container
        results = {
            'pdf_file': pdf_path,
            'total_pages': total_pages,
            'pages': [],
            'processing_summary': {
                'text_based_pages': 0,
                'image_based_pages': 0,
                'failed_pages': 0,
                'total_content_items': 0,
                'total_tables': 0,
                'total_text_blocks': 0
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Process each page
        for page_num in range(1, total_pages + 1):
            print(f"\n📑 Processing page {page_num}/{total_pages}")
            
            # Detect page type
            page_type = self.detector.detect_page_type(pdf_path, page_num)
            print(f"   📊 Detected type: {page_type.upper()}")
            
            # Process based on type
            if page_type == 'text':
                page_result = self.text_processor.process_page(pdf_path, page_num)
                if page_result['success']:
                    results['processing_summary']['text_based_pages'] += 1
            else:
                page_result = self.image_processor.process_page(pdf_path, page_num)
                if page_result['success']:
                    results['processing_summary']['image_based_pages'] += 1
            
            # Add to results
            if page_result['success']:
                results['pages'].append(page_result)
                results['processing_summary']['total_content_items'] += len(page_result.get('content', []))
                results['processing_summary']['total_tables'] += len(page_result.get('tables', []))
                results['processing_summary']['total_text_blocks'] += len(page_result.get('text_blocks', []))
                print(f"   ✅ Success: {len(page_result.get('content', []))} content items extracted")
            else:
                results['processing_summary']['failed_pages'] += 1
                results['pages'].append(page_result)
                print(f"   ❌ Failed: {page_result.get('error', 'Unknown error')}")
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir)
        
        # Print summary
        self._print_processing_summary(results)
        
        return results
    
    def process_single_page(self, pdf_path, page_number, output_dir=None):
        """
        Process a single page from a PDF
        
        Args:
            pdf_path (str): Path to PDF file
            page_number (int): Page number (1-based)
            output_dir (str): Optional output directory
            
        Returns:
            dict: Page processing results
        """
        print(f"\n📄 Processing single page {page_number} from {Path(pdf_path).name}")
        
        # Detect page type
        page_type = self.detector.detect_page_type(pdf_path, page_number)
        print(f"   📊 Detected type: {page_type.upper()}")
        
        # Process based on type
        if page_type == 'text':
            result = self.text_processor.process_page(pdf_path, page_number)
        else:
            result = self.image_processor.process_page(pdf_path, page_number)
        
        # Save single page result if output directory specified
        if output_dir and result['success']:
            os.makedirs(output_dir, exist_ok=True)
            pdf_name = Path(pdf_path).stem
            output_file = os.path.join(output_dir, f"{pdf_name}_page_{page_number}.json")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Page result saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving page result: {e}")
        
        return result
    
    def analyze_pdf_composition(self, pdf_path):
        """
        Analyze PDF composition without full processing
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: Composition analysis
        """
        print(f"\n📊 Analyzing PDF composition: {Path(pdf_path).name}")
        
        analysis = self.detector.analyze_pdf_composition(pdf_path)
        
        if analysis:
            print(f"\n📈 Composition Analysis:")
            print(f"  Total pages: {analysis['total_pages']}")
            print(f"  Text-based pages: {analysis['text_based_pages']} ({analysis['composition_ratio']['text_percentage']:.1f}%)")
            print(f"  Image-based pages: {analysis['image_based_pages']} ({analysis['composition_ratio']['image_percentage']:.1f}%)")
            
            # Recommend processing strategy
            text_ratio = analysis['composition_ratio']['text_percentage']
            if text_ratio > 80:
                strategy = "Mostly text-based - fast processing expected"
            elif text_ratio > 50:
                strategy = "Mixed document - hybrid processing optimal"
            elif text_ratio > 20:
                strategy = "Mostly image-based - OCR processing will dominate"
            else:
                strategy = "Fully scanned document - OCR processing only"
            
            print(f"  Recommended strategy: {strategy}")
        
        return analysis
    
    def _save_results(self, results, output_dir):
        """Save processing results to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_name = Path(results['pdf_file']).stem
        output_file = os.path.join(output_dir, f"{pdf_name}_hybrid_extraction.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
            
            # Also save summary
            summary = MetricsCalculator.create_processing_summary(results)
            summary_file = os.path.join(output_dir, f"{pdf_name}_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"💾 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _print_processing_summary(self, results):
        """Print processing summary"""
        summary = results['processing_summary']
        
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total pages: {results['total_pages']}")
        print(f"Text-based pages: {summary['text_based_pages']}")
        print(f"Image-based pages: {summary['image_based_pages']}")
        print(f"Failed pages: {summary['failed_pages']}")
        print(f"Total content items: {summary['total_content_items']}")
        print(f"Total tables: {summary['total_tables']}")
        print(f"Total text blocks: {summary['total_text_blocks']}")
        
        # Calculate success rate
        success_rate = ((summary['text_based_pages'] + summary['image_based_pages']) / results['total_pages'] * 100) if results['total_pages'] > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        print(f"{'='*60}")
    
    def get_processing_capabilities(self):
        """
        Get information about available processing capabilities
        
        Returns:
            dict: Capability information
        """
        capabilities = {
            'text_processing': {
                'available': hasattr(self.text_processor, 'extractor') and self.text_processor.extractor is not None,
                'method': 'PyMuPDF + layout_lm1' if hasattr(self.text_processor, 'extractor') and self.text_processor.extractor else 'PyMuPDF fallback'
            },
            'image_processing': {
                'available': True,  # Always try, but may fail if dependencies missing
                'method': 'OpenCV + Tesseract OCR'
            },
            'page_detection': {
                'available': True,
                'method': 'PyMuPDF text analysis'
            }
        }
        
        return capabilities
