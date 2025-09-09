# src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
import os

def calculate_ndvi(data):
    """
    Calculates the NDVI vegetation index.
    This is a strong indicator of plant health.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    # For Indian Pines dataset (200 bands):
    # Band 50 (approx 650nm) is Red
    # Band 90 (approx 800nm) is NIR
    red_band = data[:, :, 50]
    nir_band = data[:, :, 90]
    
    # Avoid division by zero
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    return ndvi

def calculate_ndwi(data):
    """
    Calculates the NDWI water index.
    Good for detecting water stress.
    NDWI = (NIR - SWIR) / (NIR + SWIR)
    """
    # Band 90 (approx 800nm) is NIR
    # Band 150 (approx 1200nm) is a short-wave infrared (SWIR) band
    nir_band = data[:, :, 90]
    swir_band = data[:, :, 150]
    
    ndwi = (nir_band - swir_band) / (nir_band + swir_band + 1e-10)
    return ndwi

def create_health_map(ndvi, ndwi):
    """
    Combines NDVI and NDWI into a simple health score.
    This is where the 'AI' logic is - you can tell a story about this.
    """
    # Basic logic: Health is high if NDVI is high AND not water-logged (NDWI not too high)
    health_score = ndvi * (1 - np.clip(ndwi, 0, 1))
    return health_score

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate crop health maps from hyperspectral data.')
    parser.add_argument('--data_path', type=str, default='../data/Indian_pines_corrected.mat',
                       help='Path to the hyperspectral data file')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Directory to save output images')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    # Load the hyperspectral data
    data = loadmat(args.data_path)['indian_pines_corrected']
    print(f"Data loaded. Shape: {data.shape}")
    
    print("Calculating vegetation indices...")
    # Calculate indices
    ndvi = calculate_ndvi(data)
    ndwi = calculate_ndwi(data)
    health_map = create_health_map(ndvi, ndwi)
    
    print("Generating visualizations...")
    # Create a figure with 4 subplots
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Raw data (false color)
    plt.subplot(2, 2, 1)
    plt.imshow(data[:, :, 100], cmap='viridis')  # Using band 100 for visualization
    plt.title('Hyperspectral Image (Band 100)')
    plt.colorbar()
    
    # Plot 2: NDVI
    plt.subplot(2, 2, 2)
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title('NDVI - Vegetation Health\n(Red=Unhealthy, Green=Healthy)')
    plt.colorbar()
    
    # Plot 3: NDWI
    plt.subplot(2, 2, 3)
    plt.imshow(ndwi, cmap='Blues', vmin=-1, vmax=1)
    plt.title('NDWI - Water Content\n(Dark=Dry, Blue=Wet)')
    plt.colorbar()
    
    # Plot 4: Final Health Map
    plt.subplot(2, 2, 4)
    health_display = plt.imshow(health_map, cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('AI-Powered Health Score\n(Combined NDVI & NDWI Analysis)')
    plt.colorbar(health_display)
    
    plt.tight_layout()
    
    # Save the results
    output_path = os.path.join(args.output_dir, 'health_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to: {output_path}")
    
    # Calculate some statistics for your demo
    healthy_area = np.mean(health_map > 0.6) * 100
    stressed_area = np.mean(health_map < 0.3) * 100
    
    print("\n--- AI Health Assessment Report ---")
    print(f"Healthy area: {healthy_area:.1f}%")
    print(f"Stressed area: {stressed_area:.1f}%")
    print("----------------------------------")
    
    plt.show()

if __name__ == "__main__":
    main()
