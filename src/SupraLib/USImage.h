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

#ifndef __USIMAGE_H__
#define __USIMAGE_H__

#include <memory>
#include <stddef.h>

#include "Container.h"
#include "RecordObject.h"
#include "USImageProperties.h"
#include "vec.h"

namespace supra
{
	/// A compute graph object that represents an ultrasound image with elements of type ElementType.
	/// Can contain 2D and 3D images and both before scansconversion (that is samples aling scanlines) and
	/// images / volumes after scanconversion are supported
	template <typename ElementType>
	class USImage : public RecordObject
	{
	public:
		USImage() : m_dimensions(0), m_size{}, m_pData(nullptr) {};
		/// Constructs a 3D Image
		USImage(size_t size1,
			size_t size2,
			size_t size3,
			std::shared_ptr<Container<ElementType> > pData,
			std::shared_ptr<const USImageProperties> pImageProperties,
			double receiveTimestamp,
			double syncTimestamp)
			: RecordObject(receiveTimestamp, syncTimestamp)
			, m_pImageProperties(pImageProperties)
			, m_dimensions(3)
			, m_size{ size1, size2, size3 }
			, m_pData(pData)
			, m_channels(1) {};
		/// Constructs a 2D Image
		USImage(size_t size1,
			size_t size2,
			std::shared_ptr<Container<ElementType> > pData,
			std::shared_ptr<const USImageProperties> pImageProperties,
			double receiveTimestamp, double syncTimestamp)
			: RecordObject(receiveTimestamp, syncTimestamp)
			, m_pImageProperties(pImageProperties)
			, m_dimensions(2)
			, m_size{ size1, size2, 1 }
			, m_pData(pData)
			, m_channels(1) {};
		/// Constructs a 3D or 2D Image.
		/// If size.z == 1, the image is 2D.
		USImage(vec3s size,
			std::shared_ptr<Container<ElementType> > pData,
			std::shared_ptr<const USImageProperties> pImageProperties,
			double receiveTimestamp, double syncTimestamp, size_t channels = 1)
			: RecordObject(receiveTimestamp, syncTimestamp)
			, m_pImageProperties(pImageProperties)
			, m_dimensions(3)
			, m_size(size)
			, m_pData(pData)
			, m_channels(channels)
		{
			if (size.z == 1)
			{
				m_dimensions = 2;
			}
		};
		/// Constructs a 2D Image
		USImage(vec2s size,
			std::shared_ptr<Container<ElementType> > pData,
			std::shared_ptr<const USImageProperties> pImageProperties,
			double receiveTimestamp,
			double syncTimestamp)
			: RecordObject(receiveTimestamp, syncTimestamp)
			, m_pImageProperties(pImageProperties)
			, m_dimensions(2)
			, m_pData(pData)
			, m_channels(1)
		{
			m_size.x = size.x;
			m_size.y = size.y;
			m_size.z = 1;
		};
		/// Copy constructor. Copies image metadata and the pointer to the data.
		USImage(const USImage& a)
			: RecordObject(a)
			, m_dimensions(a.m_dimensions)
			, m_size(a.m_size)
			, m_pData(a.m_pData)
			, m_pImageProperties(a.m_pImageProperties)
			, m_channels(a.m_channels) {};
		/// Special copy constructor, copies image metadata from the given image,
		/// but uses the given Container for data
		USImage(const USImage& a, std::shared_ptr<Container<ElementType> > pData)
			: RecordObject(a)
			, m_dimensions(a.m_dimensions)
			, m_size(a.m_size)
			, m_pData(pData)
			, m_pImageProperties(a.m_pImageProperties)
			, m_channels(a.m_channels) {};

		//~USImage();
		/// Returns a pointer to the \see USImageProperties that contain the associated metadata
		std::shared_ptr<const USImageProperties> getImageProperties() const { return m_pImageProperties; };
		/// Sets the \see USImageProperties that contain the associated metadata
		void setImageProperties(std::shared_ptr<USImageProperties> & imageProperties) { m_pImageProperties = imageProperties; };
		/// Returns the size of the image. If it is 2D, `getSize().z == 1`
		vec3s getSize() const { return m_size; };
		/// Returns the number of channels for this image
		size_t getNumChannels() const { return m_channels; };
		/// Returns a pointer to the image data
		std::shared_ptr<const Container<ElementType> > getData() const { return m_pData; };

		virtual RecordObjectType getType() const { return TypeUSImage; }

	private:
		std::shared_ptr<const USImageProperties> m_pImageProperties;

		std::shared_ptr<Container<ElementType> > m_pData;

		//number of image dimensions
		int m_dimensions;
		//the size of the image buffer (i.e. m_pData->size() == prod(m_size) * m_channels)
		vec3s m_size;
		//number of image channels
		size_t m_channels;
	};
}

#endif //!__USIMAGE_H__
