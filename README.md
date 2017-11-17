![SUPRA Logo](http://campar.in.tum.de/files/goeblr/supra_logo_full_small.png "SUPRA Logo")

SUPRA: Open Source Software Defined Ultrasound Processing for Real-Time Applications
================

By the [Chair for Computer Aided Medical Procedures](http://campar.in.tum.de/)

[![TUM](http://campar.in.tum.de/files/goeblr/TUM_Web_Logo_blau.png "TUM Logo")](http://tum.de)

Main contributors: 

* R&uuml;diger G&ouml;bl
* Dr. Christoph Hennersperger

Supported by [EDEN2020](http://eden2020.eu)

[![EDEN2020 Logo](http://campar.in.tum.de/files/goeblr/EDEN2020_Logo_Small.jpg "EDEN2020 Logo")](http://eden2020.eu)


A 2D and 3D Pipeline from Beamforming to B-mode
----------------

**SUPRA** is an open-source pipeline for fully software 
defined ultrasound processing for real-time applications.
Covering everything from beamforming to output of B-Mode images, SUPRA
can help reproducibility of results and allows modifications to the image acquisition.

Including all processing stages of a usual ultrasound pipeline, it can be executed in 2D and 3D on consumer GPUs in real-
time. Even on hardware as small as the CUDA enabled Jetson TX2 **SUPRA** can be run for 2D imaging in real-time.

![Standard ultrasound pipeline and where the processing takes place. Transmit beamforming is performed on the CPU, transmit and receive are performed in specialized hardware. All other processing steps (receive beamforming, envelope detection, log-compression, scan-conversion) happen in software and on the GPU](http://campar.in.tum.de/files/goeblr/UsPipeline_small.png "Standard pipeline and where the processing takes place")

License
----------------
LGPL v3
see [LICENSE](LICENSE)

Publication
----------------
If you use SUPRA for your research, please cite our work
https://arxiv.org/abs/1711.06127

G&ouml;bl, R. and Navab, N. and Hennersperger, C., SUPRA: Open Source Software Defined Ultrasound Processing for Real-Time Applications, eprint arXiv:1711.06127, Nov 2017

	@ARTICLE{2017goeblArxiv,
	   author = {{G{\"o}bl}, R. and {Navab}, N. and {Hennersperger}, C.},
		title = "{{SUPRA}: Open Source {S}oftware Defined {U}ltrasound {P}rocessing for {R}eal-Time {A}pplications}",
	  journal = {ArXiv e-prints},
	archivePrefix = "arXiv",
	   eprint = {1711.06127},
	 primaryClass = "cs.CV",
	 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Distributed, Parallel, and Cluster Computing},
		 year = 2017,
		month = nov
	}

Building
----------------
### Requirements

* cmake &ge; 3.4
* gcc &ge; 4.8 or min. Visual Studio 2015
* QT &ge; 5.5
* TBB
* CUDA &ge; 7.0
	
	
### Build instructions (Ubuntu 16.04 / 17.10)

Only 17.10:
Install GCC 6 (or any other version lower), as 6.3 is latest supported by CUDA 9.0
	
	sudo apt-get install gcc-6 g++-6

Install CMake (&ge; 3.4):	
(installs 3.5 or up on current 16.04, and 3.9 on 17.10)

	sudo apt-get install cmake cmake-gui

Install QT dev libraries (&ge; 5.5):
	
	sudo apt-get install qt5-default

Install Intel Thread Building Blocks

	sudo apt-get install libtbb-dev

Install CUDA (&ge; 7.0) as described by NVIDIA https://developer.nvidia.com/cuda-downloads .
Keep in mind that the C++ host compiler has to be supported by the CUDA version.
(Check http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html and http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html for details.)

Build OpenIGTLink (*OPTIONAL*, but recommended)

	sudo apt-get install git
	mkdir -p $HOME/git && cd $HOME/git #(or your favorite directory for repositories)
	git clone https://github.com/openigtlink/OpenIGTLink.git
	cd OpenIGTLink
	git checkout release-2.1
	mkdir -p build && cd build
	cmake -D BUILD_TESTING=OFF ..
	make -j5
	cd $HOME/git
	
SUPRA

	sudo apt-get install git
	mkdir -p $HOME/git && cd $HOME/git #(or your favorite directory for repositories)
	git clone https://github.com/IFL-CAMP/supra.git
	cd supra
	mkdir -p build && cd build
	cmake-gui ..
	
1. Configure
2. For systems with multiple gcc versions, make sure to select one supported by the installed CUDA version
3. You might need to specify the CUDA toolkit directory (usually "`/usr/local/cuda`")
4. Specify OpenIGTLink build directory for OpenIGTLink_DIR
5. Configure & Generate, then close cmake and build
6. Build SUPRA
	
	make -j5
	
7. Start SUPRA: See below
	
Demo (No US-system required!)
----------------

Change to your build directory. If you used the commands above, you can execute

	cd $HOME/git/supra/build

Start the SUPRA GUI with a demo config file

	src/GraphicInterface/SUPRA_GUI -c data/configDemo.xml -a
	
Where `-c` defines the config file to load and `-a` is autostart.

This shows a complete ultrasound pipeline running on your computer from raw channel data recorded with
a Cephasonics system and a 7MHz linear probe.
With the dropdown menu "Preview Node", you can select which stage of the pipeline to inspect.
For the final state of the image, select "SCAN", which shows the output of the scan-converter - the B-mode.
	
Used libraries
----------------

**SUPRA** uses tinyxml2 which is awesome and distributed under the zlib-license. For more details see the [tinyxml2 README](src/SupraLib/utilities/tinyxml2/readme.md) and (http://grinninglizard.com/tinyxml2/index.html and https://github.com/leethomason/tinyxml2)

On windows, ROS-message headers generated with [rosserial](http://wiki.ros.org/rosserial) are used and are included in the source.
On Linux, the usual ROS-libraries are used during build. (roscpp, geometry_msgs)

**SUPRA** additionally uses the Intel Thread Building Blocks (but does not provide them) in their Apache 2.0 licensed form. https://www.threadingbuildingblocks.org/

Finally, it can be built against
	
* QT (LGPLv3)
* IGTL (BSD 3clause)
* CAMPVis (Apache 2.0) (unfortunately, the respective QT5 version is not yet public)

Acknowledgement
----------------

![EU flag](http://campar.in.tum.de/files/goeblr/EUflag.png "EU flag")

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 688279
