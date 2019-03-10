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

#include "utility.h"

#include<chrono>
#include<fstream>

using namespace std;

namespace supra
{
	double getCurrentTime() {
		typedef chrono::high_resolution_clock clock;
		typedef chrono::duration<double> duration;

		clock::time_point curTimePoint = clock::now();
		double currentTime = chrono::duration_cast<duration>(curTimePoint.time_since_epoch()).count();

		return currentTime;
	}

	void busyWait(size_t microseconds)
	{
		typedef chrono::high_resolution_clock clock;
		typedef chrono::duration<size_t, micro> duration;
		clock::time_point start = clock::now();

		size_t diff = 0;
		do {
			diff = chrono::duration_cast<duration>(clock::now() - start).count();
		} while (diff < microseconds);
	}

	template <>
	std::string stringify(std::vector<bool> v)
	{
		std::string b;
		if (v.size() > 0)
		{
			b += "[";
			for (bool value : v)
			{
				b += (value ? "1" : "0");
			}
			b.erase(b.end() - 1, b.end());
			b += "]";
			return b;
		}
		else {
			return "[]";
		}
	}

	bool fileExists(const std::string& path)
    {
        return std::ifstream(path).good();
    }
}