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

#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <cctype>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <stddef.h>
#include <cstring>
#include <fstream>
#include <array>

namespace std
{
	/// overload of std function to ease type handling
	inline std::string to_string(std::string s)
	{
		return s;
	}
}

namespace supra
{
	using std::to_string;

#ifndef M_PI
	/// Definition of pi for the cuda compile path, as cuda math.h does not seem to provide it.
	constexpr auto M_PI = 3.14159265358979323846;
	constexpr auto M_PIx2 = 3.14159265358979323846;
#endif //!M_PI
	/// Definition of eps following the value of matlabs "eps()"
	constexpr double M_EPS = 2.2204e-16;

	/// Writes a buffer of given length as text to a file with the given filename
	template <typename T>
	void writeAscii(std::string filename, const T* buffer, size_t length)
	{
		std::ofstream o(filename);
		for (size_t i = 0; i < length; i++)
		{
			o << buffer[i] << '\n';
		}
		o.close();
	}

	template <typename T>
	void readChunks(std::ifstream& f, T* destination, size_t numElements, size_t chunkSize)
	{
		size_t numElementsChunk = chunkSize / sizeof(T);
		for (size_t elementsRead = 0; elementsRead < numElements; 
				elementsRead += numElementsChunk, destination += numElementsChunk)
		{
			size_t numToRead = std::min(chunkSize, (numElements - elementsRead) * sizeof(T));
			f.read(reinterpret_cast<char*>(destination), numToRead);
		}
	}

	/// returns the square of x
	template <typename T>
	constexpr T sq(T x)
	{
		return x*x;
	}

	/// Conversion function from degree to radian
	template <typename T>
	constexpr T degToRad(const T &deg)
	{
		return deg*M_PI / 180.0;
	}
	/// Conversion function from radian to degree
	template <typename T>
	constexpr T radToDeg(const T &rad)
	{
		return rad * 180 / M_PI;
	}

	/// Performs a copy between the given buffers while transposing the 2D-matrix,
	/// exchanging width and height
	template <typename T>
	void memcpyTransposed(T* dest, const T* src, size_t width, size_t height)
	{
		for (size_t x = 0; x < width; x++)
		{
			for (size_t y = 0; y < height; y++)
			{
				dest[y*width + x] = src[x*height + y];
			}
		}
	}

	/// Converts the argument to a string with its operator<<
	/// As opposed to stdlibs to_string the locale can be modified conviniently
	template <typename valueType>
	std::string stringify(valueType v)
	{
		std::stringstream ss;
		ss << v;
		return ss.str();
	}

	/// Converts the vector argument to a string
	template <typename valueType>
	std::string stringify(std::vector<valueType> v)
	{
		std::string b;
		if (v.size() > 0)
		{
			b += "[";
			for (valueType& value : v)
			{
				b += to_string(value) + ", ";
			}
			b.erase(b.end() - 1, b.end());
			b += "]";
			return b;
		}
		else {
			return "[]";
		}
	}

	/// Converts the argument to a string (true|false)
	template <>
	std::string stringify(std::vector<bool> v);

	/// Converts the array argument to a string
	template <typename valueType, size_t N>
	std::string stringify(std::array<valueType, N> v)
	{
		std::string b;
		if (N > 0)
		{
			b += "[";
			for (valueType& value : v)
			{
				b += to_string(value) + ", ";
			}
			b.erase(b.end() - 1, b.end());
			b += "]";
			return b;
		}
		else {
			return "[]";
		}
	}

	/// Converts the string argument to the type specified via its operator>>
	template <typename T>
	T from_string(const std::string& s) {
		std::stringstream ss(s);
		T t;
		ss >> t;
		return t;
	}

	/// Converts a string to a newly allocated cstr.
	/// ATTENTION: The returned buffer has to be deleted with `delete[] <ptr>`!
	inline char* stringToNewCstr(std::string org)
	{
		size_t len = org.length();
		char* ret = new char[len + 1];
		memcpy(ret, org.c_str(), (len + 1) * sizeof(char));
		return ret;
	}

	/// Trims leading and trailing whitespace from the string
	inline std::string trim(const std::string &str)
	{
		auto strStart = std::find_if(str.begin(), str.end(), [](int ch) {return !std::isspace(ch);} );
		auto strEnd = std::find_if(str.rbegin(), str.rend(), [](int ch) {return !std::isspace(ch);} ).base();
		if (strEnd <= strStart)
		{
			return std::string();
		}
		else {
			return std::string(strStart, strEnd);
		}
	}

	inline std::vector<std::string> split(const std::string &str, char delimiter)
	{
		std::vector<std::string> tokens;
		std::stringstream s(str);
		std::string token;
		while (std::getline(s, token, delimiter))
		{
			tokens.push_back(token);
		}
		return tokens;
	}

	/// Returns current time in seconds. The resolution depends on the operating system.
	double getCurrentTime();

	/// Performs a busy wait for the given number of microseconds. This is just a debug-tool
	void busyWait(size_t microseconds);
}

#endif // !__UTILITY_H__
