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

#include <assert.h>
#include "CampvisPreviewReciever.h"

#include "USImage.h"
#include <utilities/Logging.h>

// needed for creating OpenGL-capable windows
#include <ext/cgt/glcontextmanager.h>
#include <ext/cgt/texturereadertga.h>

// (de)initialization functions for CAMPVis
#include <core/init.h>

// main data structures
#include <core/datastructures/imagedata.h>
#include <core/eventhandlers/mwheeltonumericpropertyeventlistener.h>
#include <core/datastructures/genericimagerepresentationlocal.h>

// pipeline and processors
#include <core/pipeline/autoevaluationpipeline.h>
#include <modules/base/processors/trackballcameraprovider.h>
#include <modules/base/processors/lightsourceprovider.h>
#include <modules/vis/processors/volumeexplorer.h>

#include <core/classification/geometry1dtransferfunction.h>
#include <core/classification/tfgeometry1d.h>
#include "cgt/opengljobprocessor.h"

using namespace std;

namespace supra
{
	std::unique_ptr<cgt::QtThreadedCanvas> CampvisPreviewReciever::m_campvisBackgroundCanvas = nullptr;
	std::atomic<bool> CampvisPreviewReciever::m_campvisInitialized(false);
}

namespace campvis {
	/// CAMPVis pipeline that performs actual visualization of the volume
	class VolumeExplorerPipeline : public AutoEvaluationPipeline {
	public:
		/**
		* Creates a VolumeExplorerDemo pipeline.
		* \param	dc			Reference to the DataContainer containing local working set of data
		*           			for this pipeline, must be valid the whole lifetime of this pipeline.
		* \param	dataName 	Name of the volume to be visualized in the data container
		*/
		explicit VolumeExplorerPipeline(DataContainer& dc, const std::string & dataName)
			: AutoEvaluationPipeline(dc, getId())
			, _lsp()
			, _ve(&_canvasSize)
			, _name(dataName)
		{
			addProcessor(&_lsp);
			addProcessor(&_ve);

			addEventListenerToBack(&_ve);
		}

		/**
		* Virtual Destructor
		**/
		virtual ~VolumeExplorerPipeline() {}

		/// \see AutoEvaluationPipeline::init()
		virtual void init()
		{
			AutoEvaluationPipeline::init();

			_ve.p_outputImage.setValue("combine");
			_renderTargetID.setValue("combine");

			_ve.p_inputVolume.setValue(_name);

			Geometry1DTransferFunction* dvrTF = new Geometry1DTransferFunction(512, cgt::vec2(0.f, .05f));
			dvrTF->addGeometry(TFGeometry1D::createQuad(cgt::vec2(.12f, .15f), cgt::col4(85, 0, 0, 128), cgt::col4(255, 0, 0, 128)));
			dvrTF->addGeometry(TFGeometry1D::createQuad(cgt::vec2(.19f, .28f), cgt::col4(89, 89, 89, 155), cgt::col4(89, 89, 89, 155)));
			dvrTF->addGeometry(TFGeometry1D::createQuad(cgt::vec2(.41f, .51f), cgt::col4(170, 170, 128, 64), cgt::col4(192, 192, 128, 64)));
			static_cast<TransferFunctionProperty*>(_ve.getNestedProperty("VolumeRendererProperties::RaycasterProps::TransferFunction"))->replaceTF(dvrTF);
			static_cast<TransferFunctionProperty*>(_ve.getNestedProperty("VolumeRendererProperties::RaycasterProps::TransferFunction"))->setAutoFitWindowToData(false);
			static_cast<TransferFunctionProperty*>(_ve.getNestedProperty("VolumeRendererProperties::RaycasterProps::TransferFunction"))->setAutoFitWindowToData(false);

			static_cast<BoolProperty*>(_ve.getNestedProperty("VolumeRendererProperties::RaycasterProps::EnableShading"))->setValue(false);
			static_cast<IntProperty*>(_ve.getNestedProperty("VolumeRendererProperties::RaycasterProps::ShadowSteps"))->setValue(7);
			
			static_cast<FloatProperty*>(_ve.getNestedProperty("VolumeRendererProperties::RaycasterProps::SamplingRate"))->setValue(4.f);
			static_cast<TransferFunctionProperty*>(_ve.getNestedProperty("SliceExtractorProperties::TransferFunction"))->setAutoFitWindowToData(false);
		}

		/// \see AutoEvaluationPipeline::deinit()
		virtual void deinit() {
			AutoEvaluationPipeline::deinit();
		}

		/// \see AbstractPipeline::getName()
		virtual std::string getName() const { return getId(); };
		/// \see AbstractPipeline::getId()
		static const std::string getId() { return "VolumeExplorerPipeline"; };

	protected:
		/// Name of the volume to be visualized in the data container
		std::string _name;
		/// The LightSourceProvider for the volume renderer
		LightSourceProvider _lsp;
		/// The volume-explorer used for visualization of the volume
		VolumeExplorer _ve;
	};
}

namespace supra
{
	CampvisPreviewReciever::CampvisPreviewReciever(QWidget* parentWidget, QVBoxLayout* targetLayout, const std::string& name)
		: QWidget(parentWidget)
		, m_name(name)
	{
		for (int i = 0; i < m_name.length(); ++i) {
			if (m_name[i] == ' ')
				m_name[i] = '_';
		}

		initializeCampvis();

		// Now create and initialize the pipeline from above
		// Create a CAMPVis DataContainer where we store the images
		m_campvisDatacontainer = std::unique_ptr<campvis::DataContainer>(new campvis::DataContainer("Main DataContainer"));

		// Instantiate the pipeline we declared above
		m_pipeline = unique_ptr<campvis::VolumeExplorerPipeline>(new campvis::VolumeExplorerPipeline(*m_campvisDatacontainer, m_name));

		// Our pipeline needs a window to show the visualization, we create it here and then tell CAMPVis about it
		m_canvas = new cgt::QtThreadedCanvas("window", cgt::ivec2(cgt::GLCanvas::DEFAULT_WINDOW_WIDTH, cgt::GLCanvas::DEFAULT_WINDOW_HEIGHT),
			cgt::GLCanvas::RGBADD,
			this);
		m_canvas->init();
		cgt::GlContextManager::getRef().registerContextAndInitGlew(m_canvas, m_pipeline->getName());

		// assign the canvas to our pipeline, initialize the pipeline and enable it
		cgt::TextureReaderTga trt;
		cgt::Texture*_errorTexture = trt.loadTexture(ShdrMgr.completePath("application/data/no_input.tga"), cgt::Texture::LINEAR);
		m_pipeline->setCanvas(m_canvas);
		m_pipeline->getPipelinePainter()->setErrorTexture(_errorTexture);
		m_pipeline->init();

		// finish the initialization and start the pipeline (it is automatically run in a separate thread)
		GLCtxtMgr.releaseContext(m_canvas, false);
		// The pipeline now runs automatically in a separate thread
		campvis::startOpenGlThreadAndMoveQtThreadAffinity(m_pipeline.get(), m_canvas);

		m_canvas->setVisible(false);
		targetLayout->addWidget(m_canvas);
	}


	void CampvisPreviewReciever::previewReadyImage(const std::shared_ptr<RecordObject> image)
	{
		if (image && image->getType() == TypeUSImage)
		{
			auto inImage8Bit = std::dynamic_pointer_cast<USImage<uint8_t>>(image);
			auto inImage16Bit = std::dynamic_pointer_cast<USImage<int16_t>>(image);

			logging::log_error_if(!inImage8Bit && !inImage16Bit, "Error casting a record object to USImage, although the type was 'TypeUSImage'");
			if (inImage8Bit || inImage16Bit)
			{
				vec3s s;
				if (inImage8Bit)
				{
					s = inImage8Bit->getSize();
				}
				else {
					s = inImage16Bit->getSize();
				}
				// create a new CAMPVis image of the right size
				auto referenceImageCampvis = new campvis::ImageData(3, cgt::svec3(s.x, s.y, s.z), 1);
				// create a campvis::GenericImageRepresentationItk to convert the ITK images to CAMPVis
				if (inImage8Bit)
				{
					//TODO maybe use persistent data here (look at campvis-oct branch)
					auto image = campvis::GenericImageRepresentationLocal<uint8_t, 1>::create(referenceImageCampvis, inImage8Bit->getData()->getCopyHostRaw());
					// add the image to the DataContainer
					m_campvisDatacontainer->addData(m_name, referenceImageCampvis);
				}
				else {
					auto image = campvis::GenericImageRepresentationLocal<int16_t, 1>::create(referenceImageCampvis, inImage16Bit->getData()->getCopyHostRaw());
					// add the image to the DataContainer
					m_campvisDatacontainer->addData(m_name, referenceImageCampvis);
				}

				if (!m_pipeline->getEnabled())
				{
					m_canvas->setVisible(true);
					m_pipeline->setEnabled(true);
					for (auto it = m_pipeline->getProcessors().begin(); it != m_pipeline->getProcessors().end(); ++it) {
						(*it)->invalidate(campvis::AbstractProcessor::INVALID_RESULT);
					}
				}
			}
		}
	}

	CampvisPreviewReciever::~CampvisPreviewReciever()
	{
		m_pipeline->setEnabled(false);

		m_pipeline->stop();
	
		GLJobProc.enqueueJobBlocking([&]() {
			// Deinit pipeline first
			m_pipeline->deinit();
		});

		m_canvas->setParent(nullptr);
		delete m_canvas;
	}

	void CampvisPreviewReciever::initializeCampvis()
	{
		if (!m_campvisInitialized)
		{
			m_campvisInitialized = true;

			// make sure that the CAMPVis source directory is in the search paths so that all Shaders are found
			std::vector<std::string> searchPaths;
	#ifdef CAMPVIS_SOURCE_DIR
			searchPaths.push_back(CAMPVIS_SOURCE_DIR);
	#endif

			// CAMPVis requires a background OpenGL context, we create it here
			m_campvisBackgroundCanvas = std::unique_ptr<cgt::QtThreadedCanvas>(new cgt::QtThreadedCanvas("background"));
			// initialize CAMPVis
			campvis::init(m_campvisBackgroundCanvas.get(), searchPaths);
		}
	}
}
