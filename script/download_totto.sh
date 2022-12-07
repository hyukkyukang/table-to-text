#! /bin/bash 
# Note: Run the script from the project directory

# Directory names
Download_Dir=dataset
Evaluation_Dir=evaluation

# Download dataset
if [ -d "${Download_Dir}" ]; then
    echo "Download directory already exists! Skipping dataset setting!"
else
    echo "Setting Totto dataset..."
    mkdir -p ${Download_Dir}
    cd ${Download_Dir}
    wget https://storage.googleapis.com/totto-public/totto_data.zip
    unzip totto_data.zip && rm totto_data.zip
    cd ..
fi

# Download evaluation script
if [ -d "${Evaluation_Dir}" ]; then
    echo "evaluation directory already exists! Skipping evaluation setting!"
else
    echo "Setting evaluation scripts..."
    mkdir -p ${Evaluation_Dir}
    cd ${Evaluation_Dir}
    git clone https://github.com/google-research/language.git language_repo
    cd ..
fi

echo "Done!"