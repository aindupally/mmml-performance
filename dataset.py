#making dataset
#attempt 3 to try and make this better
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load images from a folder
def load_images_from_folder(folder):
    images = {}
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(subdir, file)
                img = plt.imread(img_path)
                img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
                skin_name = os.path.basename(subdir)
                images[f"{skin_name}/{file}"] = img
    return images

# Load performance data from CSV files
def load_performance_data(csv_folder):
    performance_data = {}
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            csv_path = os.path.join(csv_folder, file)
            skin_name = os.path.splitext(file)[0]  # e.g., skin_1 from skin_1.csv
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                output_num = f"{skin_name}/geometry_{int(row['OutputNumber'])}.jpg"
                performance_data[output_num] = row[['RibDepth', 'vM Stress (MPa)', 'Geometry Mass (kg)', 'Dir Def (mm)']].values
    return performance_data

# Load cross-section data from CSV files
def load_cross_section_data(csv_folder):
    cross_section_data = {}
    for subdir, _, files in os.walk(csv_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                skin_name = os.path.basename(subdir)
                geometry_name = os.path.splitext(file)[0]
                df = pd.read_csv(csv_path)

                # Apply the 10/10/80 sampling method
                x_data = df['X'].values
                z_data = df['Z'].values

                if len(x_data) >= 10:
                    # Sample 10% from the start, 10% from the end, and 80% from the middle
                    ten_percent = max(1, len(x_data) // 10)  # Ensure at least one sample
                    sampled_x = np.concatenate([
                        x_data[:ten_percent],  # First 10%
                        x_data[-ten_percent:],  # Last 10%
                        x_data[ten_percent: -ten_percent][::(len(x_data) - 2 * ten_percent) // 8 or 1]  # Middle 80%
                    ])

                    sampled_z = np.concatenate([
                        z_data[:ten_percent],
                        z_data[-ten_percent:],
                        z_data[ten_percent: -ten_percent][::(len(z_data) - 2 * ten_percent) // 8 or 1]
                    ])
                else:
                    # If less than 10, just use the available data
                    sampled_x = x_data
                    sampled_z = z_data

                # Combine X and Z into a single array
                combined_data = np.concatenate((sampled_x, sampled_z))  # Create a 1D array
                cross_section_data[f"{skin_name}/{geometry_name}.jpg"] = combined_data.flatten()

    return cross_section_data

# Prepare and save the dataset
def prepare_and_save_dataset(folder1, folder2, csv_folder, cross_section_folder1, cross_section_folder2, output_dir):
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)
    performance_data = load_performance_data(csv_folder)
    cross_section_data1 = load_cross_section_data(cross_section_folder1)
    cross_section_data2 = load_cross_section_data(cross_section_folder2)

    print(f"Number of images in folder1: {len(images1)}")
    print(f"Number of images in folder2: {len(images2)}")
    print(f"Number of performance data entries: {len(performance_data)}")
    print(f"Number of cross-section data entries from folder 1: {len(cross_section_data1)}")
    print(f"Number of cross-section data entries from folder 2: {len(cross_section_data2)}")

    X_top_view, X_side_view, X_depth, X_cross_25, X_cross_75, y = [], [], [], [], [], []
    
    for key in images1:
        if key in images2 and key in performance_data:
            X_top_view.append(images1[key])
            X_side_view.append(images2[key])
            X_depth.append(performance_data[key][0])  # Depth data

            # Cross-section data from both folders
            if key in cross_section_data1:
                padded_cross_section1 = pad_sequences([cross_section_data1[key]], maxlen=600, padding='post', dtype='float32')[0]
                X_cross_25.append(padded_cross_section1)
            else:
                X_cross_25.append(np.zeros(600, dtype='float32'))  # Add zero padding if missing

            if key in cross_section_data2:
                padded_cross_section2 = pad_sequences([cross_section_data2[key]], maxlen=600, padding='post', dtype='float32')[0]
                X_cross_75.append(padded_cross_section2)
            else:
                X_cross_75.append(np.zeros(600, dtype='float32'))  # Add zero padding if missing

            y.append(performance_data[key][1:])  # Performance data: Stress, Mass, Deflection

    try:
        X_top_view, X_side_view, X_depth, X_cross_25, X_cross_75, y = np.array(X_top_view), np.array(X_side_view), np.array(X_depth), np.array(X_cross_25), np.array(X_cross_75), np.array(y)
    except Exception as e:
        print(f"Error converting to numpy arrays: {e}")
        return

    # Create validation dataset
    indices = np.random.choice(len(X_top_view), size=20, replace=False)  # Randomly select 20 indices for validation
    X_top_view_val = X_top_view[indices]
    X_side_view_val = X_side_view[indices]
    X_depth_val = X_depth[indices]
    X_cross_25_val = X_cross_25[indices]
    X_cross_75_val = X_cross_75[indices]
    y_val = y[indices]

    # Create training dataset by excluding validation entries
    mask = np.ones(len(X_top_view), dtype=bool)
    mask[indices] = False
    X_top_view_train, X_side_view_train, X_depth_train, X_cross_25_train, X_cross_75_train, y_train = (
        X_top_view[mask],
        X_side_view[mask],
        X_depth[mask],
        X_cross_25[mask],
        X_cross_75[mask],
        y[mask]
    )

    # Save the data with more readable keys
    np.savez(os.path.join(output_dir, 'training_dataset.npz'),
             top_view_images=X_top_view_train,
             side_view_images=X_side_view_train,
             depth_data=X_depth_train,
             cross_section_25=X_cross_25_train,
             cross_section_75=X_cross_75_train,
             performance_data=y_train)

    np.savez(os.path.join(output_dir, 'validation_dataset.npz'),
             top_view_images=X_top_view_val,
             side_view_images=X_side_view_val,
             depth_data=X_depth_val,
             cross_section_25=X_cross_25_val,
             cross_section_75=X_cross_75_val,
             performance_data=y_val)

# Main execution
folder1 = r'F:\2024-10-09 Multi Modal Model Data\2024-11-15 128x128 top view'  # Path for top view images
folder2 = r'F:\2024-10-09 Multi Modal Model Data\2024-11-15 128x128 side view'  # Path for side view images
csv_folder = r'F:\2024-10-09 Multi Modal Model Data\2024-10-15 Performance and Depth Data csvs'  # Path for performance data
cross_section_folder1 = r'F:\2024-10-09 Multi Modal Model Data\2024-11-04 Cleaned 25% Cross Sections'  # Path for cross-section data 1
cross_section_folder2 = r'F:\2024-10-09 Multi Modal Model Data\2024-11-04 Cleaned 75% Cross Sections'  # Path for cross-section data 2
output_dir = r'F:\2024-11-04 Trained Models, Training and Validation Data\2024-11-04 Training and Validation Data'  # Path to save training and validation data

# Run the data preparation and saving
prepare_and_save_dataset(folder1, folder2, csv_folder, cross_section_folder1, cross_section_folder2, output_dir)

print(f"Data collection complete. Training dataset saved to {output_dir}/training_dataset.npz and validation dataset saved to {output_dir}/validation_dataset.npz")
