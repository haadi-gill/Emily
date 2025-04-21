from PIL import Image
import os
import numpy as np

# Input and output directories
input_root = 'project_data'
output_root = 'GSPics'

# Create the output root if needed
if not os.path.exists(output_root):
    os.makedirs(output_root)

print("Converting images in:", input_root)

data = []
labels = []

# Get a sorted list of subdirectories in the input_root (backpack, shoes, etc.)
class_names = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

# Loop through each subdirectory (backpack, shoes, etc.)
for class_index, class_name in enumerate(class_names):
    class_input_path = os.path.join(input_root, class_name)
    class_output_path = os.path.join(output_root, class_name)

    if not os.path.exists(class_output_path):
        os.makedirs(class_output_path)

    for file in os.listdir(class_input_path):
        if file.lower().endswith('.png'):
            input_file_path = os.path.join(class_input_path, file)
            output_file_path = os.path.join(class_output_path, file)

            # Convert image to grayscale
            img = Image.open(input_file_path).convert('L')
            img.save(output_file_path)

            # Save data and label
            img_np = np.asarray(img)
            data.append(img_np.reshape(-1))
            labels.append(class_index)  # Use the folder index as the label

            print(input_file_path)

# Convert lists to arrays
data = np.array(data).astype(int)
labels = np.array(labels).astype(int)

# Save CSVs to output_root
np.savetxt(os.path.join(output_root, "data_array.csv"), data, delimiter=",", fmt='%i')
np.savetxt(os.path.join(output_root, "labels_array.csv"), labels, delimiter=",", fmt='%i')

print(f"Converted {len(labels)} images successfully")
print("Done!")
