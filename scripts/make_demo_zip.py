"""Zips data/new_data_demo (folder per class) so you can upload it in the dashboard during the demo."""
import shutil
shutil.make_archive("new_data_demo", "zip", "data", "new_data_demo")
print("Created new_data_demo.zip")
