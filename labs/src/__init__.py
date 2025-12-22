"""
Chess Stylometry Analysis - Source Modules
===========================================

This package contains the core modules for chess stylometry analysis:
- pipeline_stylometry_blocks: Main pipeline for block-based analysis
- multi_channel_model: Multi-channel CNN model builder
- multi_channel_generator: Data generators for training
- generate_images_blocks: Image generation for blocks
- generate_decision_heatmaps: Heatmap generation
- extract_player_games_by_event_parallel: Player game extraction
- event_discovery_parallel: Event discovery in PGN files
- parse_games_to_images: Game parsing and image overlay
"""

__all__ = [
    'pipeline_stylometry_blocks',
    'multi_channel_model',
    'multi_channel_generator',
    'generate_images_blocks',
    'generate_decision_heatmaps',
    'extract_player_games_by_event_parallel',
    'event_discovery_parallel',
    'parse_games_to_images'
]
