DOCKERID=$1
sudo docker stop $DOCKERID && sudo docker rm $DOCKERID && sudo docker run -d -p 6006:6006 -v $(pwd)/logs:/logs --name my-tf-tensorboard volnet/tensorflow-tensorboard