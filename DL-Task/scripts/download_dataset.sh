# create a directory to store the dataset
mkdir -p ./data

echo "Downloading the dataset..."
wget https://cchsu.info/files/images.zip

# unzip the dataset into the directory
unzip images.zip -d data