import os

# Directory path
dir_path = "/Users/aswinkumarv/Desktop/video_length/result_align"

# Loop through all files in the directory
for filename in os.listdir(dir_path):
    # Check if the file is an '.align' file
    if filename.endswith(".align"):
        # Full file path
        file_path = os.path.join(dir_path, filename)

        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify each line
        modified_lines = []
        for line in lines:
            words = line.split()  # Split by whitespace
            if len(words) == 2:
                # Swap the order of the words and prepend '0 '
                modified_line = f"0 {words[1]} {words[0]}\n"
                modified_lines.append(modified_line)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

print("File modification complete!")

