import os
import random
import kagglehub

print("Downloading ADE20K dataset via kagglehub...")
# This automatically downloads to the system cache and returns the path
kaggle_path = kagglehub.dataset_download("awsaf49/ade20k-dataset")
print(f"Dataset downloaded to: {kaggle_path}")

# Kaggle extracts the folder directly, but let's ensure we are pointing to the right root
ade_root = os.path.join(kaggle_path, "ADEChallengeData2016")
if not os.path.exists(ade_root):
    ade_root = kaggle_path

train_img_dir = os.path.join(ade_root, "images", "training")
val_img_dir = os.path.join(ade_root, "images", "validation")

# Get all filenames (mmsegmentation expects names without the .jpg extension)
train_files = [f.replace('.jpg', '') for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
val_files = [f.replace('.jpg', '') for f in os.listdir(val_img_dir) if f.endswith('.jpg')]

print(f"Found {len(train_files)} training images and {len(val_files)} validation images.")

# Step 1: Create a symlink in the project directory
project_data_dir = "data/ade/ADEChallengeData2016"
os.makedirs("data/ade", exist_ok=True)

if not os.path.exists(project_data_dir):
    os.symlink(ade_root, project_data_dir)
    print(f"Created symlink: {project_data_dir} -> {ade_root}")
else:
    print(f"Symlink already exists at {project_data_dir}")

# Step 2: Create the 10% and 50% Data Splits
print("Generating 10% and 50% split files...")
random.seed(42) # Ensure reproducible splits!
random.shuffle(train_files)

num_train = len(train_files)
train_10 = train_files[:int(num_train * 0.1)]
train_50 = train_files[:int(num_train * 0.5)]

# Save the split lists directly into the linked project folder
def save_split(filename, file_list):
    with open(os.path.join(project_data_dir, filename), 'w') as f:
        for name in file_list:
            f.write(f"{name}\n")

save_split("train_100.txt", train_files)
save_split("train_50.txt", train_50)
save_split("train_10.txt", train_10)
save_split("val.txt", val_files)

print("Done! Your D3 data pipeline is linked and split files are ready.")