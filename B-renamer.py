import os
preprocessed_folder = "../R0008026-postprocessed"
"""A temporary folder used to export the pre-processed files: greyscale, and with limited empty images."""

for filename in os.listdir(preprocessed_folder):
    parts = os.path.splitext(filename)
    fnameParts = parts[0].split("-")
    os.rename(
        os.path.join(preprocessed_folder, filename),
        os.path.join(preprocessed_folder, str.join("-",fnameParts[:-1]) + parts[-1])
    )
