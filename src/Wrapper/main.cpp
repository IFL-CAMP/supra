// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include <utilities/Logging.h>

#include <SupraManager.h>

using namespace supra;

int main(int argc, char** argv) {
	logging::Base::setLogLevel(logging::info);

	if (argc == 1)
	{
		logging::log_always("Usage: SUPRA_CMD <config.xml>");
	}
	else if (argc >= 2)
	{
		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::milliseconds milliseconds;
		Clock::time_point t0 = Clock::now();
		Clock::time_point t00 = Clock::now();

		SupraManager::Get()->readFromXml(argv[1]);

		Clock::time_point t1 = Clock::now();
		milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		//std::cout << "Time for reading XML: " << ms.count() << "ms\n";

		t0 = Clock::now();
		SupraManager::Get()->startOutputs();
		SupraManager::Get()->startInputs();
		t1 = Clock::now();

		ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		//std::cout << "Time for start outputs and start inputs: " << ms.count() << "ms\n";

		t0 = Clock::now();
		SupraManager::Get()->stopAndWaitInputs();
		t1 = Clock::now();

		ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		//std::cout << "Time for stopandWaitinputs: " << ms.count() << "ms\n";

		t0 = Clock::now();
		SupraManager::Get()->waitForGraph();
		t1 = Clock::now();

		ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
		//std::cout << "Time for waitForGraph: " << ms.count() << "ms\n";
		
		Clock::time_point t2 = Clock::now();
		ms = std::chrono::duration_cast<milliseconds>(t2 - t00);
		//std::cout << "Time for Reconstruction: " << ms.count() << "ms\n";
	
		//std::cout << "all done" << std::endl;
	}
}
