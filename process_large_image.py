#!/usr/bin/env python3
"""
Standalone Script for Processing Large Satellite Images

This script can be used to preprocess large satellite images before loading them
into the Streamlit app. It produces downsampled versions that are easier to work with.

Usage:
python process_large_image.py input.img output.img

Options:
  --scale SCALE   Downsample factor (default: 8)
  --window SIZE   Processing window size (default: 128)
  --help          Show this help message
"""

import os
import sys
import argparse
import rasterio
import numpy as np
from skimage.measure import block_reduce
import gc

def force_garbage_collection():
    """Force garbage collection to free memory"""
    for _ in range(3):
        gc.collect()

def downsample_large_image(input_path, output_path, scale=8, window_size=128):
    """
    Downsample a large satellite image to make it more manageable.
    
    Args:
        input_path: Path to input image
        output_path: Path to save downsampled image
        scale: Downsample factor
        window_size: Processing window size
    """
    print(f"Processing {input_path}")
    print(f"Output will be saved to {output_path}")
    
    # Open the source file to get metadata
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        out_width = src.width // scale
        out_height = src.height // scale
        
        # Create output profile
        profile = src.profile.copy()
        profile.update({
            'width': out_width,
            'height': out_height,
            'transform': rasterio.Affine(
                src.transform.a * scale,
                src.transform.b,
                src.transform.c,
                src.transform.d,
                src.transform.e * scale,
                src.transform.f
            )
        })
        
        # Create output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Process each band
            for band_idx in range(1, src.count + 1):
                print(f"Processing band {band_idx} of {src.count}")
                
                # Create output array
                out_data = np.zeros((out_height, out_width), dtype=profile['dtype'])
                
                # Process the image in tiles
                for y in range(0, src.height, window_size * scale):
                    for x in range(0, src.width, window_size * scale):
                        # Calculate window dimensions
                        win_width = min(window_size * scale, src.width - x)
                        win_height = min(window_size * scale, src.height - y)
                        
                        # Skip small edge windows
                        if win_width < scale or win_height < scale:
                            continue
                            
                        # Read window
                        window = rasterio.windows.Window(x, y, win_width, win_height)
                        data = src.read(band_idx, window=window)
                        
                        # Handle nodata
                        if src.nodata is not None:
                            data[data == src.nodata] = 0  # Replace nodata values with 0