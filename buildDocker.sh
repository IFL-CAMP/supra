#/bin/bash

echo "CURRENTLY THIS DOCKER IS ONLY FOR BUILDING ON:"
echo "  * DGX, UBUNTU 16.04, CUDA 9.2"
echo "  * Xavier, L4T, UBUNTU 18.04, CUDA 10.2 (with txdocker)"

ARCH="$(uname -m)"
echo "arch: ${ARCH}"

IMAGENAME=test
TAG=nvsupra

mkdir -p build_docker

# build the docker image with the required build environment
if [ $ARCH = "aarch64" ]
then
	sudo docker build -t $IMAGENAME/cudaxavier -f dockerfiles/Dockerfile_cudaxavier/Dockerfile_cudaxavier dockerfiles/Dockerfile_cudaxavier
	sudo docker build -t $IMAGENAME/${TAG}_builder -f dockerfiles/Dockerfile_buildEnv_Xavier .
	DOCKERCMD=txdocker
else
	sudo docker build -t $IMAGENAME/${TAG}_builder -f dockerfiles/Dockerfile_buildEnv_DGX .
	DOCKERCMD=docker
fi

# build the software in a build container
sudo $DOCKERCMD run -it -v `pwd`/build_docker:/nvSupra/build $IMAGENAME/${TAG}_builder

# create run-time container
if [ $ARCH = "aarch64" ]
then
	cp dockerfiles/Dockerfile_Xavier build_docker/Dockerfile
else
	cp dockerfiles/Dockerfile_DGX build_docker/Dockerfile
fi
sudo docker build -t $IMAGENAME/${TAG} build_docker

echo " "
echo "docker built."

echo " "
echo "To run the REST interface execute the following command:"
echo "sudo $DOCKERCMD run -it -p 6502:6502 -p 18944:18944 -p 18945:18945 -v <host data folder>:/data ${IMAGENAME}/${TAG} /nvSupra/build/src/RestInterface/SUPRA_REST /data/configDemo.xml"
