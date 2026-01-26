import os
import numpy as np
from stl import mesh
from PIL import Image
import io

import matplotlib.pyplot as plt

def load_stl(filepath):
    """Load STL file and return mesh object."""
    return mesh.Mesh.from_file(filepath)

def center_and_project(mesh_obj, view='top'):
    """Center mesh and project to 2D based on view."""
    vertices = mesh_obj.vectors.reshape(-1, 3)
    
    # Center the geometry
    center = vertices.mean(axis=0)
    vertices = vertices - center
    
    # Project based on view
    if view == 'top':
        projection = vertices[:, :2]  # x, y
    elif view == 'side':
        projection = vertices[:, [0, 2]]  # x, z
    
    return projection

def crop_to_square(projection, size=128):
    """Crop projection to square and normalize."""
    # Get bounds
    min_coord = projection.min(axis=0)
    max_coord = projection.max(axis=0)
    
    # Create square bounds
    range_coord = max_coord - min_coord
    max_range = range_coord.max()
    center = (min_coord + max_coord) / 2
    
    # Normalize to [-1, 1]
    normalized = (projection - center) / (max_range / 2 + 1e-6)
    
    return normalized

def projection_to_image(projection, size=128):
    """Convert 2D projection to image."""
    # Generate at higher resolution for better quality
    high_res = size * 4  # 4x resolution before downsampling
    
    fig, ax = plt.subplots(figsize=(high_res/100, high_res/100), dpi=100)
    ax.scatter(projection[:, 0], projection[:, 1], s=1, c='black')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    img = Image.open(buf).convert('L')
    img = img.resize((high_res, high_res), Image.Resampling.LANCZOS)
    # Scale down to final size for better anti-aliasing
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img)

def process_stl_directory(directory, output_dir='./output', size=128):
    """Process all STL files in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.stl'):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            
            try:
                mesh_obj = load_stl(filepath)
                
                # Generate top view
                top_proj = center_and_project(mesh_obj, 'top')
                top_proj = crop_to_square(top_proj, size)
                top_img = projection_to_image(top_proj, size)
                print(f"Generated top view for {filename}")
                
                # Generate side view
                side_proj = center_and_project(mesh_obj, 'side')
                side_proj = crop_to_square(side_proj, size)
                side_img = projection_to_image(side_proj, size)
                print(f"Generated side view for {filename}")
                
                # Save images
                base_name = os.path.splitext(filename)[0]
                Image.fromarray(top_img).save(os.path.join(output_dir, f'{base_name}_top.png'))
                Image.fromarray(side_img).save(os.path.join(output_dir, f'{base_name}_side.png'))
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    stl_directory = 'data/ch10k/'  # Change to your STL directory
    process_stl_directory(stl_directory)