source vertex_config.env

docker build -t $IMAGE_URI -f Dockerfile .

docker push $IMAGE_URI
