"""
Metrics Calculator
Calculates quality and performance metrics for PDF processing
"""

import numpy as np
from datetime import datetime


class MetricsCalculator:
    """
    Calculates various metrics for PDF extraction quality assessment
    """
    
    @staticmethod
    def calculate_text_based_metrics(text_blocks, entities, tables, image_size):
        """
        Calculate metrics for text-based processing
        
        Args:
            text_blocks (list): List of extracted text blocks
            entities (list): List of extracted entities
            tables (list): List of extracted tables
            image_size (tuple): Size of the processed image/page
            
        Returns:
            dict: Comprehensive metrics
        """
        metrics = {}
        
        # Basic extraction metrics
        metrics['total_text_blocks'] = len(text_blocks)
        metrics['total_entities'] = len(entities)
        metrics['total_tables'] = len(tables)
        
        # Text coverage metrics
        total_text_length = sum(len(block.get('text', '')) for block in text_blocks)
        metrics['total_text_length'] = total_text_length
        
        # Entity distribution metrics
        entity_types = {}
        confidence_scores = []
        
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            confidence = entity.get('confidence', 1.0)
            
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            confidence_scores.append(confidence)
        
        metrics['entity_types_count'] = len(entity_types)
        metrics['avg_confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
        metrics['min_confidence'] = min(confidence_scores) if confidence_scores else 1.0
        metrics['max_confidence'] = max(confidence_scores) if confidence_scores else 1.0
        
        # Layout quality metrics
        if text_blocks:
            block_sizes = [len(block.get('text', '')) for block in text_blocks]
            metrics['avg_block_size'] = np.mean(block_sizes)
            metrics['median_block_size'] = np.median(block_sizes)
        else:
            metrics['avg_block_size'] = 0
            metrics['median_block_size'] = 0
        
        # Coverage estimation (approximate)
        if image_size and text_blocks:
            page_area = image_size[0] * image_size[1]
            text_area_estimate = total_text_length * 100  # Rough estimate
            metrics['text_coverage_ratio'] = min(1.0, text_area_estimate / page_area)
        else:
            metrics['text_coverage_ratio'] = 0.0
        
        # Quality score calculation (0-100)
        quality_score = MetricsCalculator._calculate_quality_score(
            total_text_length, len(entities), len(entity_types), 
            metrics['avg_confidence'], len(tables)
        )
        
        metrics['quality_score'] = quality_score
        metrics['accuracy_percentage'] = quality_score  # Using quality score as accuracy approximation
        metrics['processing_method'] = 'text_based'
        
        # Entity type breakdown
        for entity_type, count in entity_types.items():
            metrics[f'entity_{entity_type}_count'] = count
        
        return metrics
    
    @staticmethod
    def calculate_image_based_metrics(content_items):
        """
        Calculate metrics for image-based (OCR) processing
        
        Args:
            content_items (list): List of extracted content items
            
        Returns:
            dict: OCR processing metrics
        """
        metrics = {}
        
        # Basic extraction metrics
        text_blocks = [item for item in content_items if item.get('type') == 'text']
        tables = [item for item in content_items if item.get('type') == 'table']
        
        metrics['total_text_blocks'] = len(text_blocks)
        metrics['total_tables'] = len(tables)
        metrics['total_content_items'] = len(content_items)
        
        # Text and confidence metrics
        total_text_length = sum(len(item.get('content', '')) for item in text_blocks)
        confidence_scores = [item.get('confidence', 0) for item in content_items if 'confidence' in item]
        
        metrics['total_text_length'] = total_text_length
        metrics['avg_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
        metrics['min_confidence'] = min(confidence_scores) if confidence_scores else 0.0
        metrics['max_confidence'] = max(confidence_scores) if confidence_scores else 0.0
        
        # OCR quality assessment
        high_conf_items = [conf for conf in confidence_scores if conf > 0.8]
        med_conf_items = [conf for conf in confidence_scores if 0.5 < conf <= 0.8]
        low_conf_items = [conf for conf in confidence_scores if conf <= 0.5]
        
        metrics['high_confidence_items'] = len(high_conf_items)
        metrics['medium_confidence_items'] = len(med_conf_items)
        metrics['low_confidence_items'] = len(low_conf_items)
        
        # Quality score for OCR
        avg_conf = metrics['avg_confidence']
        quality_score = min(100, avg_conf * 100)
        
        metrics['quality_score'] = quality_score
        metrics['accuracy_percentage'] = quality_score
        metrics['processing_method'] = 'image_based_ocr'
        
        return metrics
    
    @staticmethod
    def _calculate_quality_score(text_length, entity_count, entity_types, avg_confidence, table_count):
        """
        Calculate an overall quality score (0-100)
        """
        quality_score = 0
        
        # Text extraction quality (40 points)
        if text_length > 1000:
            quality_score += 40
        elif text_length > 500:
            quality_score += 30
        elif text_length > 100:
            quality_score += 20
        elif text_length > 10:
            quality_score += 10
        
        # Entity detection quality (30 points)
        if entity_count > 10:
            quality_score += 30
        elif entity_count > 5:
            quality_score += 20
        elif entity_count > 2:
            quality_score += 10
        elif entity_count > 0:
            quality_score += 5
        
        # Confidence quality (20 points)
        if avg_confidence > 0.9:
            quality_score += 20
        elif avg_confidence > 0.8:
            quality_score += 15
        elif avg_confidence > 0.6:
            quality_score += 10
        elif avg_confidence > 0.4:
            quality_score += 5
        
        # Structure detection quality (10 points)
        if table_count > 0:
            quality_score += 5
        if entity_types > 3:
            quality_score += 5
        
        return min(100, quality_score)
    
    @staticmethod
    def create_processing_summary(results):
        """
        Create a comprehensive processing summary
        
        Args:
            results (dict): Processing results from hybrid processor
            
        Returns:
            dict: Summary statistics and insights
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pages': results.get('total_pages', 0),
            'processing_breakdown': results.get('processing_summary', {}),
            'success_rate': 0,
            'avg_quality_score': 0,
            'avg_confidence': 0,
            'content_distribution': {
                'total_text_blocks': 0,
                'total_tables': 0,
                'total_content_items': 0
            },
            'method_performance': {
                'text_based_avg_quality': 0,
                'image_based_avg_quality': 0
            }
        }
        
        # Calculate success rate
        successful_pages = (
            results.get('processing_summary', {}).get('text_based_pages', 0) +
            results.get('processing_summary', {}).get('image_based_pages', 0)
        )
        total_pages = results.get('total_pages', 0)
        summary['success_rate'] = (successful_pages / total_pages * 100) if total_pages > 0 else 0
        
        # Aggregate quality metrics
        pages = results.get('pages', [])
        successful_pages_data = [p for p in pages if p.get('success', False)]
        
        if successful_pages_data:
            quality_scores = [p.get('metrics', {}).get('quality_score', 0) for p in successful_pages_data]
            confidence_scores = [p.get('metrics', {}).get('avg_confidence', 0) for p in successful_pages_data]
            
            summary['avg_quality_score'] = np.mean(quality_scores)
            summary['avg_confidence'] = np.mean(confidence_scores)
            
            # Method-specific performance
            text_based_pages = [p for p in successful_pages_data if p.get('processing_method') == 'text_based']
            image_based_pages = [p for p in successful_pages_data if p.get('processing_method') == 'image_based']
            
            if text_based_pages:
                text_quality = [p.get('metrics', {}).get('quality_score', 0) for p in text_based_pages]
                summary['method_performance']['text_based_avg_quality'] = np.mean(text_quality)
            
            if image_based_pages:
                image_quality = [p.get('metrics', {}).get('quality_score', 0) for p in image_based_pages]
                summary['method_performance']['image_based_avg_quality'] = np.mean(image_quality)
            
            # Content distribution
            summary['content_distribution']['total_text_blocks'] = sum(
                len(p.get('text_blocks', [])) for p in successful_pages_data
            )
            summary['content_distribution']['total_tables'] = sum(
                len(p.get('tables', [])) for p in successful_pages_data
            )
            summary['content_distribution']['total_content_items'] = sum(
                len(p.get('content', [])) for p in successful_pages_data
            )
        
        return summary
