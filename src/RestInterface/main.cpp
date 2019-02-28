/////// BASED ON: https://github.com/ivanmejiarocha/micro-service
/////// AND SUPRA
//
//  Created by Ivan Mejia on 12/24/16.
//
// MIT License
//
// Copyright (c) 2016 ivmeroLabs.
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <utilities/Logging.h>

#include <SupraManager.h>

#include <iostream>

#include <usr_interrupt_handler.hpp>
#include <runtime_utils.hpp>

#include "RestInterface.h"

using namespace supra;

using namespace web;
using namespace cfx;

int main(int argc, const char * argv[]) {
	
	logging::Base::setLogLevel(
		logging::info |
		logging::warning |
		logging::error |
		logging::param |
		logging::external);

	if(argc < 2)
	{
		logging::log_always("Usage: SUPRA_REST <config.xml>");
	}
	else 
	{
		SupraManager::Get()->setFreezeTimeout(36000);
		SupraManager::Get()->readFromXml(argv[1]);

		SupraManager::Get()->startOutputs();
		SupraManager::Get()->startInputs();

		//Interface
		{	
			InterruptHandler::hookSIGINT();

			RestInterface server;
			server.setEndpoint("http://host_auto_ip4:6502/");
			
			try {
				// wait for server initialization...
				server.accept().wait();
				std::cout << "Modern C++ Microservice now listening for requests at: " << server.endpoint() << '\n';
				std::cout << "Usage: eg. " << server.endpoint() << "nodes/all" << '\n';
				
				InterruptHandler::waitForUserInterrupt();

				server.shutdown().wait();
			}
			catch(std::exception & e) {
				std::cerr << "somehitng wrong happen! :(" << '\n';
			}
			catch(...) {
				RuntimeUtils::printStackTrace();
			}
		}

		//stop inputs
		SupraManager::Get()->stopAndWaitInputs();

		//wait for remaining messages to be processed
		SupraManager::Get()->waitForGraph();
	}
	
    return 0;
}
