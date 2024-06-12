# make the directory
mkdir -p ./checkpoints

# set the model file path
WEIGHT_FILE=./checkpoints/pretrained_weight.pth
weight_url="https://ncku365-my.sharepoint.com/:u:/g/personal/p76121233_ncku_edu_tw/EbjHC6UT2NpKojo477DQx3MBtgO1lrg9qwxJcFMIAYaUTA?e=OF5IuZ&download=1"

# download the pretrained weight
echo "Downloading the pretrained weight ..."
wget -N $weight_url -O $WEIGHT_FILE