"""
Advanced Usage Examples for PDF Hybrid Processing Utility
Demonstrates batch processing, custom configurations, and integration patterns
"""

import sys
import os
from pathlib import Path
import json

# Add the utility to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pdf_hybrid_utility import HybridPDFProcessor


def batch_processing_example():
    """
    Example: Process multiple PDFs in batch
    """
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    processor = HybridPDFProcessor(
        text_threshold=30,      # Lower threshold for more text detection
        confidence_threshold=60 # Lower confidence for more OCR results
    )
    
    # Find all PDFs in input directories
    input_dirs = ["../input_pdfs", "../input_pdfs2"]
    pdf_files = []
    
    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            pdf_files.extend(list(Path(input_dir).glob("*.pdf")))
    
    if not pdf_files:
        print("❌ No PDF files found in input directories")
        return None
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    batch_results = []
    output_dir = "batch_outputs"
    
    for pdf_path in pdf_files[:3]:  # Process first 3 files for demo
        print(f"\n🔄 Processing: {pdf_path.name}")
        
        try:
            result = processor.process_pdf(str(pdf_path), output_dir)
            
            # Create summary for this file
            summary = {
                'file': pdf_path.name,
                'status': 'success',
                'pages': result['total_pages'],
                'text_pages': result['processing_summary']['text_based_pages'],
                'image_pages': result['processing_summary']['image_based_pages'],
                'content_items': result['processing_summary']['total_content_items'],
                'success_rate': ((result['processing_summary']['text_based_pages'] + 
                                result['processing_summary']['image_based_pages']) / 
                               result['total_pages'] * 100) if result['total_pages'] > 0 else 0
            }
            batch_results.append(summary)
            print(f"✅ Completed: {pdf_path.name}")
            
        except Exception as e:
            summary = {
                'file': pdf_path.name,
                'status': 'failed',
                'error': str(e)
            }
            batch_results.append(summary)
            print(f"❌ Failed: {pdf_path.name} - {e}")
    
    # Print batch summary
    print(f"\n📊 BATCH PROCESSING SUMMARY:")
    print(f"{'File':<25} | {'Status':<8} | {'Pages':<6} | {'Text':<4} | {'Image':<5} | {'Success%':<8}")
    print("-" * 75)
    
    for result in batch_results:
        if result['status'] == 'success':
            print(f"{result['file']:<25} | {'✅':<8} | {result['pages']:<6} | {result['text_pages']:<4} | {result['image_pages']:<5} | {result['success_rate']:<8.1f}")
        else:
            print(f"{result['file']:<25} | {'❌':<8} | {'ERROR':<6} | {'-':<4} | {'-':<5} | {'-':<8}")
    
    return batch_results


def custom_configuration_example():
    """
    Example: Using custom configurations for different scenarios
    """
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    pdf_path = "../input_pdfs2/testing6.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    # Configuration 1: High precision (strict thresholds)
    print("🎯 Configuration 1: High Precision")
    processor_precise = HybridPDFProcessor(
        text_threshold=100,     # High threshold - only clear text pages
        confidence_threshold=85 # High confidence - only best OCR results
    )
    
    try:
        result1 = processor_precise.process_single_page(pdf_path, 1, "config1_output")
        print(f"   Method used: {result1.get('processing_method', 'unknown')}")
        print(f"   Content items: {len(result1.get('content', []))}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Configuration 2: High recall (lenient thresholds)
    print("\n🌐 Configuration 2: High Recall")
    processor_lenient = HybridPDFProcessor(
        text_threshold=10,      # Low threshold - catch minimal text
        confidence_threshold=30 # Low confidence - accept more OCR results
    )
    
    try:
        result2 = processor_lenient.process_single_page(pdf_path, 1, "config2_output")
        print(f"   Method used: {result2.get('processing_method', 'unknown')}")
        print(f"   Content items: {len(result2.get('content', []))}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Configuration 3: Balanced (default)
    print("\n⚖️ Configuration 3: Balanced")
    processor_balanced = HybridPDFProcessor()  # Default settings
    
    try:
        result3 = processor_balanced.process_single_page(pdf_path, 1, "config3_output")
        print(f"   Method used: {result3.get('processing_method', 'unknown')}")
        print(f"   Content items: {len(result3.get('content', []))}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return {
        'precise': result1 if 'result1' in locals() else None,
        'lenient': result2 if 'result2' in locals() else None,
        'balanced': result3 if 'result3' in locals() else None
    }


def integration_pattern_example():
    """
    Example: Common integration patterns
    """
    print("\n" + "=" * 60)
    print("INTEGRATION PATTERN EXAMPLE")
    print("=" * 60)
    
    processor = HybridPDFProcessor()
    pdf_path = "../input_pdfs2/testing6.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    # Pattern 1: Pre-analysis before processing
    print("🔍 Pattern 1: Pre-Analysis Strategy")
    analysis = processor.analyze_pdf_composition(pdf_path)
    
    if analysis:
        image_ratio = analysis['composition_ratio']['image_percentage']
        
        if image_ratio > 70:
            print("   Strategy: Image-heavy document - allocate more OCR resources")
            # You could adjust OCR settings here
        elif image_ratio < 30:
            print("   Strategy: Text-heavy document - fast processing expected")
            # You could use optimized text processing
        else:
            print("   Strategy: Mixed document - hybrid processing optimal")
    
    # Pattern 2: Page-by-page processing with custom logic
    print("\n🔄 Pattern 2: Page-by-Page with Custom Logic")
    
    try:
        doc_results = []
        
        for page_num in range(1, min(3, analysis['total_pages'] + 1)):  # Process first 2 pages
            page_result = processor.process_single_page(pdf_path, page_num)
            
            if page_result['success']:
                # Custom business logic based on page content
                content_types = [item['type'] for item in page_result.get('content', [])]
                
                if 'table' in content_types:
                    print(f"   Page {page_num}: Contains tables - flagged for data extraction")
                    # Custom table processing logic here
                
                if page_result.get('metrics', {}).get('avg_confidence', 1.0) < 0.7:
                    print(f"   Page {page_num}: Low confidence - flagged for manual review")
                    # Quality assurance logic here
                
                doc_results.append(page_result)
            else:
                print(f"   Page {page_num}: Processing failed")
        
        print(f"   Successfully processed {len(doc_results)} pages")
        
    except Exception as e:
        print(f"   Error in page-by-page processing: {e}")
    
    # Pattern 3: Result post-processing
    print("\n🔧 Pattern 3: Result Post-Processing")
    
    try:
        # Full document processing
        full_result = processor.process_pdf(pdf_path, "integration_output")
        
        # Extract all text content
        all_text = []
        all_tables = []
        
        for page in full_result.get('pages', []):
            if page.get('success'):
                # Collect text
                for item in page.get('content', []):
                    if item['type'] == 'text':
                        all_text.append(item['content'])
                    elif item['type'] == 'table':
                        all_tables.append(item['content'])
        
        # Create consolidated output
        consolidated = {
            'document_summary': {
                'total_text_length': sum(len(text) for text in all_text),
                'total_tables': len(all_tables),
                'processing_methods_used': list(set(
                    page.get('processing_method', 'unknown') 
                    for page in full_result.get('pages', []) 
                    if page.get('success')
                ))
            },
            'full_text': '\n'.join(all_text),
            'tables': all_tables
        }
        
        # Save consolidated result
        output_file = "integration_output/consolidated_result.json"
        os.makedirs("integration_output", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        print(f"   Consolidated result saved to: {output_file}")
        print(f"   Total text length: {consolidated['document_summary']['total_text_length']}")
        print(f"   Total tables: {consolidated['document_summary']['total_tables']}")
        
        return consolidated
        
    except Exception as e:
        print(f"   Error in post-processing: {e}")
        return None


def main():
    """
    Run all advanced examples
    """
    print("🚀 PDF Hybrid Processing Utility - Advanced Examples")
    print("Demonstrates batch processing, configurations, and integration patterns\n")
    
    # Example 1: Batch processing
    batch_result = batch_processing_example()
    
    # Example 2: Custom configurations
    config_result = custom_configuration_example()
    
    # Example 3: Integration patterns
    integration_result = integration_pattern_example()
    
    print("\n" + "=" * 60)
    print("🎯 ADVANCED EXAMPLES SUMMARY")
    print("=" * 60)
    print("✅ All advanced examples completed!")
    print("📁 Check these output directories:")
    print("  - batch_outputs/ (batch processing results)")
    print("  - config1_output/ (high precision config)")
    print("  - config2_output/ (high recall config)")
    print("  - config3_output/ (balanced config)")
    print("  - integration_output/ (integration patterns)")
    
    return {
        'batch': batch_result,
        'configurations': config_result,
        'integration': integration_result
    }


if __name__ == "__main__":
    main()
