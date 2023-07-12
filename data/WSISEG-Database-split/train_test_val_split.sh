#!/bin/bash

# Set the paths to your image and annotation directories
image_dir="./whole-sky-images"
annotation_dir="./annotation"

# Set the path to the test, validation, and train directories inside the image and annotation directories
test_image_dir="$image_dir/test"
test_annotation_dir="$annotation_dir/test"
val_image_dir="$image_dir/val"
val_annotation_dir="$annotation_dir/val"
train_image_dir="$image_dir/train"
train_annotation_dir="$annotation_dir/train"

##### Step 0
# # Create the test, validation, and train directories if they don't exist
mkdir -p "$test_image_dir"
mkdir -p "$test_annotation_dir"
mkdir -p "$val_image_dir"
mkdir -p "$val_annotation_dir"
mkdir -p "$train_image_dir"
mkdir -p "$train_annotation_dir"

# Set the number of images you want to randomly select for test and validation
num_test_images=50
num_val_images=50


##### Step 1
# # Get the list of all image files
image_files=$(find "$image_dir" -maxdepth 1 -type f)

# # Randomly select 50 images for test
shuf -e "${image_files[@]}" | head -n "$num_test_images" | xargs -I {} mv {} "$test_image_dir"


##### Step 2
# Move the corresponding annotations for test to the test directory
for image_path in "$test_image_dir"/*; do
  image_name=$(basename "$image_path")
  annotation_path="$annotation_dir/$image_name"
  mv "$annotation_path" "$test_annotation_dir"/
done


##### Step 3
# Remove the moved files from the image_files array
image_files=$(find "$image_dir" -maxdepth 1 -type f)

# Randomly select 50 images for validation
echo shuf -e "${image_files[@]}" | head -n "$num_val_images" | xargs -I {} mv {} "$val_image_dir"


##### Step 4
# Move the corresponding annotations for validation to the validation directory
for image_path in "$val_image_dir"/*; do
  image_name=$(basename "$image_path")
  annotation_path="$annotation_dir/$image_name"
  mv "$annotation_path" "$val_annotation_dir"/
done


##### Step 5
find $image_dir -maxdepth 1 -type f -exec mv {} $train_image_dir  \;
find $annotation_dir -maxdepth 1 -type f -exec mv {} $train_annotation_dir  \;

# # Count the number of files in each directory for images
echo "Images Directory:"
find "$image_dir" -maxdepth 1 -type f | wc -l

echo "Test Images Directory:"
find "$test_image_dir" -maxdepth 1 -type f | wc -l

echo "Validation Images Directory:"
find "$val_image_dir" -maxdepth 1 -type f | wc -l

echo "Train Images Directory:"
find "$train_image_dir" -maxdepth 1 -type f | wc -l

# Count the number of files in each directory for annotations
echo "Annotations Directory:"
find "$annotation_dir" -maxdepth 1 -type f | wc -l

echo "Test Annotations Directory:"
find "$test_annotation_dir" -maxdepth 1 -type f | wc -l

echo "Validation Annotations Directory:"
find "$val_annotation_dir" -maxdepth 1 -type f | wc -l

echo "Train Annotations Directory:"
find "$train_annotation_dir" -maxdepth 1 -type f | wc -l
