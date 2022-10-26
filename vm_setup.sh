# Create VM and have this ready:
sudo su

cd ..

apt-get update

yes Y |  apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

yes Y | apt-get update
yes Y | apt-get install docker-ce docker-ce-cli containerd.io
# if necessary:
# VERSION=$(curl --silent https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*\d')
# DESTINATION=/usr/bin/docker-compose
# sudo curl -L https://github.com/docker/compose/releases/download/${VERSION}/docker-compose-$(uname -s)-$(uname -m) -o $DESTINATION
# sudo chmod 755 $DESTINATION


curl https://sdk.cloud.google.com > install.sh
bash install.sh --disable-prompts

exec -l $SHELL

sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose


gcloud init # Choose the service account credentials option nr 1, then specify project_id = molten-box-279222


# For whatever reason I can't run gsutil cp -R gs://biotech_lee/bert_api/ bert_api
for file in $( gsutil ls gs://mrc_python_files/bert_api/ ); do
    if [ $file != "gs://mrc_python_files/bert_api/" ]
    then gsutil cp $file ${file#"gs://mrc_python_files/bert_api/"}
    #then gsutil cp $file
    fi
done

#cd bert_api/bert_api


docker build -t docker-flask-api .

docker run -d -it -p 5000:5000 -v $(pwd):/app  docker-flask-api
