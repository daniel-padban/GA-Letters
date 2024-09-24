import zipfile

with zipfile.ZipFile('EMNIST_zip.zip','r') as zip_file:
    zip_file.printdir()
    zip_file.extractall('EMNIST_data')