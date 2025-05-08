import rasterio
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
import gc
from contextlib import contextmanager

# --- Memory Management Helpers ---
@contextmanager
def rasterio_open_with_retry(path, retries=3):
    """Context manager to handle rasterio open with retries and proper error handling"""
    for attempt in range(retries):
        try:
            src = rasterio.open(path)
            yield src
            src.close()
            break
        except rasterio.errors.RasterioIOError as e:
            if attempt == retries - 1:
                raise e
            import time
            time.sleep(1)


def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()

# --- Improved Index Computation ---
def compute_indices(image_path, window_size=512):
    """
    Compute remote sensing indices from a satellite image using memory-efficient windowed reading.
    Supports NDVI, NDBI, ExG (normalized), and VARI.
    """
    indices = { 'ndvi': None, 'ndbi': None, 'exg': None, 'vari': None }

    try:
        with rasterio_open_with_retry(image_path) as src:
            width, height = src.width, src.height
            window_size = min(window_size, width, height)
            x_off = (width - window_size) // 2
            y_off = (height - window_size) // 2
            window = rasterio.windows.Window(x_off, y_off, window_size, window_size)

            # Read bands into dict
            bands = {}
            for idx, name in enumerate(['red', 'green', 'blue', 'nir', 'swir'], start=1):
                if idx <= src.count:
                    arr = src.read(idx, window=window).astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    bands[name] = arr
            # Extract bands
            red = bands.get('red'); green = bands.get('green')
            blue = bands.get('blue'); nir = bands.get('nir'); swir = bands.get('swir')

            # NDVI: (NIR - Red) / (NIR + Red)
            if red is not None and nir is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (nir - red) / (nir + red)
                indices['ndvi'] = np.clip(ndvi, -1.0, 1.0)

            # NDBI: (SWIR - NIR) / (SWIR + NIR)
            if nir is not None and swir is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndbi = (swir - nir) / (swir + nir)
                indices['ndbi'] = np.clip(ndbi, -1.0, 1.0)

            # ExG (Normalized): (2*G - R - B) / (R + G + B)
            if red is not None and green is not None and blue is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    exg_raw = 2 * green - red - blue
                    exg_norm = exg_raw / (red + green + blue)
                indices['exg'] = np.nan_to_num(exg_norm, nan=0.0)

            # VARI: (G - R) / (G + R - B)
            if red is not None and green is not None and blue is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    vari = (green - red) / (green + red - blue)
                indices['vari'] = np.clip(vari, -1.0, 1.0)

    except Exception as e:
        print(f"Error computing indices: {e}")
    finally:
        force_garbage_collection()

    return indices

# --- Visualization Functions ---
def plot_single_index(index, title="Index Map", figsize=(4, 3), cmap='YlGn'):
    """Visualize a single index array with appropriate colormap and scale limits"""
    #fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots(figsize=(4, 4))  # Smaller image

    if index is not None:
        # Select colormap
        if title.lower() in ['ndbi', 'built-up']:
            cmap = 'RdYlBu_r'
        elif title.lower() in ['change', 'difference']:
            cmap = 'coolwarm'
        im = ax.imshow(index, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    return fig



# def plot_change_map(index1, index2, title="Change Map", figsize=(4, 4)):
#     """Visualize the difference between two index arrays with enhanced clarity and smaller size"""
#     fig, ax = plt.subplots(figsize=figsize)
#     if index1 is not None and index2 is not None:
#         diff = np.clip(index2 - index1, -1, 1)
#         vmax = np.nanpercentile(np.abs(diff), 99)
#         #im = ax.imshow(diff, cmap='seismic', vmin=-vmax, vmax=vmax)
#         vmax = np.nanpercentile(np.abs(diff), 99)
#         im = ax.imshow(diff, cmap='seismic', vmin=-vmax, vmax=vmax)

#         ax.set_title(title, fontsize=10)
#         cbar = fig.colorbar(im, ax=ax, shrink=0.7)
#         cbar.set_label('Change')
#         if 'ndvi' in title.lower():
#             ax.text(0.02, 0.02, "Red: Gain", color='darkred', transform=ax.transAxes, fontsize=7)
#             ax.text(0.02, 0.06, "Blue: Loss", color='darkblue', transform=ax.transAxes, fontsize=7)
#     else:
#         ax.text(0.5, 0.5, "No data available for comparison", ha='center', va='center', transform=ax.transAxes)
#     ax.axis('off')
#     return fig
def plot_change_map(index1, index2, title="Change Map", figsize=(1, 1)):
    """Visualize the difference between two index arrays with enhanced clarity and smaller size"""
    fig, ax = plt.subplots(figsize=figsize)  # You can reduce figsize to something like (3, 3)
    if index1 is not None and index2 is not None:
        diff = np.clip(index2 - index1, -1, 1)
        vmax = np.nanpercentile(np.abs(diff), 99)
        im = ax.imshow(diff, cmap='seismic', vmin=-vmax, vmax=vmax)

        ax.set_title(title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Change')
        if 'ndvi' in title.lower():
            ax.text(0.02, 0.02, "Red: Gain", color='darkred', transform=ax.transAxes, fontsize=7)
            ax.text(0.02, 0.06, "Blue: Loss", color='darkblue', transform=ax.transAxes, fontsize=7)
    else:
        ax.text(0.5, 0.5, "No data available for comparison", ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    return fig




# def save_fig_to_bytes(fig, dpi=60):
#     """Convert matplotlib figure to smaller PNG bytes."""
#     buf = BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, pad_inches=0.05)
#     plt.close(fig)
#     buf.seek(0)
#     return buf.read()

def save_fig_to_bytes(fig, dpi=50):  # Lower dpi from 80 to 60 or even 50 for smaller image size
    """Convert matplotlib figure to smaller PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --- Main Analysis Function ---
def analyze_image(uploaded_file):
    """Process uploaded image and return computed indices and their figures"""
    if not uploaded_file:
        return None
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.img') as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        indices = compute_indices(path)
        result = {'filename': uploaded_file.name, 'path': path}
        for key, arr in indices.items():
            if arr is not None and not np.all(np.isnan(arr)):
                result[key] = arr
                result[f"{key}_fig"] = plot_single_index(arr, title=key.upper())
        return result
    except Exception as e:
        print(f"Error analyzing {uploaded_file.name}: {e}")
        if os.path.exists(path): os.unlink(path)
        return None
    finally:
        force_garbage_collection()

