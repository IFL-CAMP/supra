nvSUPRA

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
[https://doi.org/10.1007/s11548-018-1750-6](https://doi.org/10.1007/s11548-018-1750-6)

G&ouml;bl, R., Navab, N. & Hennersperger, C. , "SUPRA: Open Source Software Defined Ultrasound Processing for Real-Time Applications" Int J CARS (2018). https://doi.org/10.1007/s11548-018-1750-6

	@Article{Goebl2018supra,
		author="G{\"o}bl, R{\"u}diger and Navab, Nassir and Hennersperger, Christoph",
		title="SUPRA: open-source software-defined ultrasound processing for real-time applications",
		journal="International Journal of Computer Assisted Radiology and Surgery",
		year="2018",
		month="Mar",
		day="28",
		issn="1861-6429",
		doi="10.1007/s11548-018-1750-6",
		url="https://doi.org/10.1007/s11548-018-1750-6"
	}
