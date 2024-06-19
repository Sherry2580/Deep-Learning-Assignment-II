# make the directory
mkdir -p ./checkpoints

# set the model file paths
WEIGHT_FILE1=./checkpoints/pretrained_weight_ImprovedCNN.pth
weight_url1="https://ncku365-my.sharepoint.com/:u:/g/personal/p76121233_ncku_edu_tw/EbjHC6UT2NpKojo477DQx3MBtgO1lrg9qwxJcFMIAYaUTA?e=AJvv2B&download=1"

WEIGHT_FILE2=./checkpoints/pretrained_weight_ComplexCNN.pth
weight_url2="&download=1"

WEIGHT_FILE3=./checkpoints/pretrained_weight_ResNet34.pth
weight_url3="https://ncku365-my.sharepoint.com/:u:/g/personal/p76121233_ncku_edu_tw/ETBh2VTeTr1IqfsczyUwLyUBZZ4u3mOKy85y3bwO1SjYmQ?e=G8cq5I&download=1"

# download the pretrained weights
echo "Downloading the pretrained weights ..."
wget -N $weight_url1 -O $WEIGHT_FILE1
wget -N $weight_url2 -O $WEIGHT_FILE2
wget -N $weight_url3 -O $WEIGHT_FILE3
