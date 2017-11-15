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
#include "RosInterface.h"

using namespace supra;

int main(int argc, char** argv) {
	logging::Base::setLogLevel(logging::info);

	if(argc < 3)
	{
		logging::log_always("Usage: SUPRA_ROS <config.xml> <ros master host/IP>");
	}
	else 
	{
		SupraManager::Get()->readFromXml(argv[1]);

		SupraManager::Get()->startOutputs();
		SupraManager::Get()->startInputs();

		//Interface
		RosInterface::mainLoop(argv[2]);

		//stop inputs
		SupraManager::Get()->stopAndWaitInputs();

		//wait for remaining messages to be processed
		SupraManager::Get()->waitForGraph();
	}
}
