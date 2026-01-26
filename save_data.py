import numpy as np
import os
from PIL import Image

def load_and_save_npz_data(npz_file_path, output_dir):
    """
    Load data from an npz file and save each array in its original format.
    
    Args:
        npz_file_path: Path to the input .npz file
        output_dir: Directory where the individual files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the npz file
    data = np.load(npz_file_path)
    
    # Save each key-value pair based on data type
    for key in data.keys():
        array = data[key]
        
        if 'image' in key.lower() or 'png' in key.lower():
            if array.dtype != np.uint8:
                array = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)

            def to_pil(img):
                img = np.squeeze(img)
                if img.ndim == 2:
                    return Image.fromarray(img, mode="L")  # grayscale
                if img.ndim == 3 and img.shape[-1] in (3, 4):
                    return Image.fromarray(img)            # RGB / RGBA
                raise ValueError(f"Unsupported image shape {img.shape} for key {key}")

            if len(array.shape) == 3:  # Single image
                output_path = os.path.join(output_dir, f"{key}.png")
                to_pil(array).save(output_path)
                print(f"Saved {key} with shape {array.shape} to {output_path}")
            elif len(array.shape) == 4:  # Multiple images
                for i, img in enumerate(array):
                    output_path = os.path.join(output_dir, f"{key}_{i}.png")
                    to_pil(img).save(output_path)
                print(f"Saved {len(array)} images from {key} to {output_dir}")
        
        elif 'cross' in key.lower() or 'section' in key.lower() or 'depth' in key.lower() or 'performance' in key.lower():
            # Save cross-sections, depth data, and performance as CSV
            output_path = os.path.join(output_dir, f"{key}.csv")
            if len(array.shape) == 1:
                np.savetxt(output_path, array.reshape(-1, 1), delimiter=',')
            else:
                np.savetxt(output_path, array.reshape(array.shape[0], -1), delimiter=',')
            print(f"Saved {key} with shape {array.shape} to {output_path}")
        
        else:
            # Save everything else as .npy
            output_path = os.path.join(output_dir, f"{key}.npy")
            np.save(output_path, array)
            print(f"Saved {key} with shape {array.shape} to {output_path}")
    
    data.close()


if __name__ == "__main__":
    # Example usage
    npz_file = "data/validation_dataset.npz"  # Replace with your npz file path
    output_directory = "data"     # Replace with desired output directory
    
    load_and_save_npz_data(npz_file, output_directory)