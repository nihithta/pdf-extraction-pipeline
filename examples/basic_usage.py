"""
Basic Usage Examples for PDF Hybrid Processing Utility
"""

import sys
import os
from pathlib import Path

# Add the utility to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pdf_hybrid_utility import HybridPDFProcessor


def basic_example():
    """
    Basic example: Process a single PDF
    """
    print("=" * 60)
    print("BASIC EXAMPLE: Single PDF Processing")
    print("=" * 60)
    
    # Initialize processor
    processor = HybridPDFProcessor(
        text_threshold=50,      # Minimum characters for text-based detection
        confidence_threshold=70 # Minimum OCR confidence
    )
    
    # Test file path - update this to your actual PDF
    pdf_path = "../input_pdfs2/testing6.pdf"
    output_dir = "basic_output"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable to point to your PDF file")
        return None
    
    try:
        # Process the PDF
        results = processor.process_pdf(pdf_path, output_dir)
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return None


def single_page_example():
    """
    Example: Process just a single page
    """
    print("\n" + "=" * 60)
    print("SINGLE PAGE EXAMPLE")
    print("=" * 60)
    
    processor = HybridPDFProcessor()
    
    pdf_path = "../input_pdfs2/testing6.pdf"
    page_number = 1
    output_dir = "single_page_output"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    try:
        # Process single page
        result = processor.process_single_page(pdf_path, page_number, output_dir)
        
        if result['success']:
            print(f"\n✅ Page {page_number} processed successfully!")
            print(f"Processing method: {result['processing_method']}")
            print(f"Content items: {len(result.get('content', []))}")
            print(f"Quality score: {result.get('metrics', {}).get('quality_score', 0):.1f}")
        else:
            print(f"❌ Page {page_number} processing failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def composition_analysis_example():
    """
    Example: Analyze PDF composition without full processing
    """
    print("\n" + "=" * 60)
    print("COMPOSITION ANALYSIS EXAMPLE")
    print("=" * 60)
    
    processor = HybridPDFProcessor()
    
    pdf_path = "../input_pdfs2/testing6.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    try:
        # Analyze composition
        analysis = processor.analyze_pdf_composition(pdf_path)
        
        if analysis:
            print("\n📊 Quick Analysis Results:")
            print(f"  Document type: {'Mixed' if analysis['text_based_pages'] > 0 and analysis['image_based_pages'] > 0 else 'Uniform'}")
            print(f"  Estimated processing time: {'Medium' if analysis['image_based_pages'] > 5 else 'Fast'}")
            print(f"  OCR pages: {analysis['image_based_pages']}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def capabilities_example():
    """
    Example: Check processing capabilities
    """
    print("\n" + "=" * 60)
    print("CAPABILITIES CHECK")
    print("=" * 60)
    
    processor = HybridPDFProcessor()
    capabilities = processor.get_processing_capabilities()
    
    print("🔍 Available Processing Capabilities:")
    
    for capability_type, info in capabilities.items():
        status = "✅ Available" if info['available'] else "❌ Not Available"
        print(f"  {capability_type.replace('_', ' ').title()}: {status}")
        print(f"    Method: {info['method']}")
    
    return capabilities


def main():
    """
    Run all examples
    """
    print("🚀 PDF Hybrid Processing Utility - Examples")
    print("Demonstrates automatic text vs image page detection\n")
    
    # Example 1: Basic processing
    basic_result = basic_example()
    
    # Example 2: Single page
    single_page_result = single_page_example()
    
    # Example 3: Composition analysis
    composition_result = composition_analysis_example()
    
    # Example 4: Capabilities check
    capabilities_result = capabilities_example()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print("✅ All examples completed!")
    print("📁 Check the output directories for results:")
    print("  - basic_output/")
    print("  - single_page_output/")
    
    return {
        'basic': basic_result,
        'single_page': single_page_result,
        'composition': composition_result,
        'capabilities': capabilities_result
    }


if __name__ == "__main__":
    main()
