#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <memory>
#include <QFileDialog>
#include <QString>
#include <QScrollBar>
#include <QTimer>
#include <QListWidget>

#include <parametersWidget.h>
#include <previewBuilderQt.h>
#include "NodeExplorerDataModel.h"

#include <utilities/utility.h>
#include <utilities/Logging.h>
#include <SupraManager.h>
#include <InterfaceFactory.h>

using namespace std;

namespace supra
{
	/// minimum width of the nodes list
	constexpr int listWidthMin = 50;

	/// Sets minimum and maximum width of a list widget to the width-hint
	void setMinMaxWidthAdaptive(QListWidget* o, bool keepMax = false)
	{
		int curW = o->minimumWidth();
		int w = std::max(o->sizeHintForColumn(0) + 2 * o->frameWidth(), listWidthMin);

		if (keepMax)
		{
			w = max(w, curW);
		}

		o->setMinimumWidth(w);
		o->setMaximumWidth(w);
	}

	MainWindow::MainWindow(QWidget *parent) :
		QMainWindow(parent),
		ui(new Ui::MainWindow),
		m_sequenceStarted(false),
		m_started(false),
		m_pParametersWidget(nullptr),
		m_preview(nullptr),
		m_previousCursorPosition(0, 0)
	{
		ui->setupUi(this);
		m_pSequenceShortcut = new QShortcut(QKeySequence("F5"), this);

		QTimer *timer = new QTimer(this);
		QTimer *timerFreeze = new QTimer(this);

		connect(ui->actionLoadConfig, SIGNAL(triggered()), this, SLOT(loadConfigFileAction()));
		connect(ui->pushButtonLoad, SIGNAL(clicked()), this, SLOT(loadConfigFileAction()));
		connect(ui->pushButtonStart, SIGNAL(clicked()), this, SLOT(startNodes()));
		connect(ui->pushButtonStop, SIGNAL(clicked()), this, SLOT(stopNodes()));
		connect(ui->pushButtonSequence, SIGNAL(clicked()), this, SLOT(startStopSequence()));
		connect(m_pSequenceShortcut, SIGNAL(activated()), this, SLOT(startStopSequence()));
		connect(ui->actionLoglevelInfo, SIGNAL(triggered()), this, SLOT(setLogLevel()));
		connect(ui->actionLoglevelLog, SIGNAL(triggered()), this, SLOT(setLogLevel()));
		connect(ui->actionLoglevelWarning, SIGNAL(triggered()), this, SLOT(setLogLevel()));
		connect(ui->actionLoglevelError, SIGNAL(triggered()), this, SLOT(setLogLevel()));
		connect(ui->actionLoglevelParameter, SIGNAL(triggered()), this, SLOT(setLogLevel()));
		connect(ui->actionLoglevelExternals, SIGNAL(triggered()), this, SLOT(setLogLevel()));

		connect(ui->actionPreviewSizeSmall, SIGNAL(triggered()), this, SLOT(setPreviewSize()));
		connect(ui->actionPreviewSizeMedium, SIGNAL(triggered()), this, SLOT(setPreviewSize()));
		connect(ui->actionPreviewSizeLarge, SIGNAL(triggered()), this, SLOT(setPreviewSize()));
		connect(ui->actionPreviewSizeWambo, SIGNAL(triggered()), this, SLOT(setPreviewSize()));
		connect(ui->actionPreviewLinearInterpolation, SIGNAL(triggered()), this, SLOT(setLinearInterpolation()));

		connect(ui->comboBoxPreviewNode, SIGNAL(currentIndexChanged(const QString &)), this, SLOT(previewSelected(const QString &)));

		connect(timer, SIGNAL(timeout()), this, SLOT(updateNodeTimings()));
		connect(timerFreeze, SIGNAL(timeout()), this, SLOT(updateFreezeTimer()));
		connect(ui->pushButtonFreeze, SIGNAL(clicked()), this, SLOT(toogleFreeze()));
		connect(ui->pushButtonResetFreeze, SIGNAL(clicked()), this, SLOT(resetFreezeTimer()));
		connect(this, &MainWindow::externClose, this, &MainWindow::close);

		p_manager = SupraManager::Get();

		m_previewSize = QSize(350, 400);
		ui->actionPreviewSizeMedium->setChecked(true);

		m_previewLinearInterpolation = true;
		ui->actionPreviewLinearInterpolation->setChecked(true);

		timer->start(2000);
		timerFreeze->start(1000);

		//ui->group_allNodes->layout
		shared_ptr<QtNodes::DataModelRegistry> flowRegistry = make_shared<QtNodes::DataModelRegistry>();
		for (auto nodeType : InterfaceFactory::getNodeTypes())
		{
			flowRegistry->registerModel(std::unique_ptr<NodeExplorerDataModel>(new NodeExplorerDataModel(nodeType, nodeType)));
		}

		m_pNodeScene = new QtNodes::FlowScene(flowRegistry);
		m_pNodeScene->setParent(this);
		m_pNodeView = new QtNodes::FlowView(m_pNodeScene, ui->group_allNodes);
		
		ui->group_allNodes->layout()->addWidget(m_pNodeView);

		connect(m_pNodeScene, &QtNodes::FlowScene::selectionChanged, this, &MainWindow::nodeSceneSelectionChanged);
		connect(m_pNodeScene, &QtNodes::FlowScene::connectionCreated, this, &MainWindow::connectionCreated);
		connect(m_pNodeScene, &QtNodes::FlowScene::connectionDeleted, this, &MainWindow::connectionDeleted);
		connect(m_pNodeScene, &QtNodes::FlowScene::nodeCreated, this, &MainWindow::nodeCreated);
		connect(m_pNodeScene, &QtNodes::FlowScene::nodeDeleted, this, &MainWindow::nodeDeleted);
	}

	MainWindow::~MainWindow()
	{
		delete ui;
	}

	void MainWindow::prepareForClose()
	{
		m_nodeSignalsDeactivated = true;
		//remove the previews
		if (m_preview)
		{
			m_preview->setParent(nullptr);
			m_preview->removePreviews();
		}
	}

	void MainWindow::quitCallback()
	{
		emit externClose();
	}

	void MainWindow::closeEvent(QCloseEvent *event)
	{
		p_manager->stopAndWaitInputs();
		p_manager->waitForGraph();

		prepareForClose();

		QMainWindow::closeEvent(event);
	}

	void MainWindow::loadConfigFileAction()
	{
		QString filename = QFileDialog::getOpenFileName(this,
			tr("Open Config XML"), "", tr("XML files (*.xml)"));
		loadConfigFile(filename);
	}

	void MainWindow::loadConfigFile(const QString & filename)
	{
		m_nodeSignalsDeactivated = true;
		p_manager->readFromXml(filename.toStdString().c_str());
		auto positions = computeNodePositions();
		vec2f gridSpacing{ 250, 100 };

		auto nodeTypes = p_manager->getNodeTypes();
		for (string node : p_manager->getNodeIDs())
		{
			auto& sceneNode = m_pNodeScene->createNode(std::unique_ptr<NodeExplorerDataModel>(new NodeExplorerDataModel(node, nodeTypes[node])));
			auto position = static_cast<vec2f>(positions[node]) * gridSpacing;
			m_pNodeScene->setNodePosition(sceneNode, QPointF(position.x, position.y));
			m_NodeSceneNodeIDs[node] = sceneNode.id();

			if (p_manager->getNode(node)->getNumOutputs() > 0)
			{
				ui->comboBoxPreviewNode->addItem(QString::fromStdString(node));
			}
		}

		m_pNodeScene->iterateOverNodes([this](QtNodes::Node* node) {
			this->m_NodeSceneNodes[node->id()] = node;
		});
		string prevNodeID = "";
		if (m_preview)
		{
			prevNodeID = m_preview->getNodeID();
		}

		for (auto connection : p_manager->getNodeConnections())
		{
			string fromNodeID = get<0>(connection);
			string toNodeID = get<2>(connection);
			if (prevNodeID == "" || toNodeID != prevNodeID)
			{
				m_pNodeScene->createConnection(
					*(m_NodeSceneNodes[m_NodeSceneNodeIDs[toNodeID]]),
					static_cast<QtNodes::PortIndex>(get<3>(connection)),
					*(m_NodeSceneNodes[m_NodeSceneNodeIDs[fromNodeID]]),
					static_cast<QtNodes::PortIndex>(get<1>(connection))
				);
			}
		}
		
		ui->pushButtonLoad->setDisabled(true);
		ui->actionLoadConfig->setDisabled(true);
		ui->pushButtonStart->setEnabled(true);		

		m_nodeSignalsDeactivated = false;
	}

	void MainWindow::keyPressEvent(QKeyEvent * keyEvent)
	{
		resetFreezeTimer();
		QMainWindow::keyPressEvent(keyEvent);
	}

	void MainWindow::showParametersFromList(QString nodeID)
	{
		if (m_pParametersWidget)
		{
			delete m_pParametersWidget;
		}

		ui->group_parameters->setTitle(QString("Parameters: ") + nodeID);

		m_pParametersWidget = new parametersWidget(nodeID, ui->group_parameters);
		ui->scrollArea_parameters->setWidget(m_pParametersWidget);
		int widthBefore = ui->scrollArea_parameters->width();
		ui->scrollArea_parameters->setMinimumWidth(0);
		ui->scrollArea_parameters->resize(10, ui->scrollArea_parameters->height());
		int scrollbarWidth = qApp->style()->pixelMetric(QStyle::PM_ScrollBarExtent);
		ui->scrollArea_parameters->setMinimumWidth(max(m_pParametersWidget->width() + scrollbarWidth, widthBefore));
	}
	void MainWindow::hideParameters()
	{
		if (m_pParametersWidget)
		{
			delete m_pParametersWidget;
			m_pParametersWidget = nullptr;
			ui->group_parameters->setTitle(QString("Parameters: Select Node"));
		}
	}

	void MainWindow::nodeSceneSelectionChanged()
	{
		auto nodes = m_pNodeScene->selectedNodes();
		if (nodes.size() >= 1)
		{
			showParametersFromList(nodes[0]->nodeDataModel()->caption());
		}
	}

	void MainWindow::connectionCreated(QtNodes::Connection & connection)
	{
		if (!m_nodeSignalsDeactivated)
		{
			auto nodeFrom = connection.getNode(QtNodes::PortType::Out);
			auto nodeTo = connection.getNode(QtNodes::PortType::In);
			if (nodeFrom && nodeTo)
			{
				string nodeFromID = nodeFrom->nodeDataModel()->caption().toStdString();
				string nodeToID = nodeTo->nodeDataModel()->caption().toStdString();
				size_t portFrom = connection.getPortIndex(QtNodes::PortType::Out);
				size_t portTo = connection.getPortIndex(QtNodes::PortType::In);
				p_manager->connect(nodeFromID, portFrom, nodeToID, portTo);
			}
		}
	}

	void MainWindow::connectionDeleted(QtNodes::Connection & connection)
	{
		if (!m_nodeSignalsDeactivated)
		{
			auto nodeFrom = connection.getNode(QtNodes::PortType::Out);
			auto nodeTo = connection.getNode(QtNodes::PortType::In);
			if (nodeFrom && nodeTo)
			{
				string nodeFromID = nodeFrom->nodeDataModel()->caption().toStdString();
				string nodeToID = nodeTo->nodeDataModel()->caption().toStdString();
				size_t portFrom = connection.getPortIndex(QtNodes::PortType::Out);
				size_t portTo = connection.getPortIndex(QtNodes::PortType::In);
				p_manager->disconnect(nodeFromID, portFrom, nodeToID, portTo);
			}
		}
	}

	void MainWindow::nodeCreated(QtNodes::Node & node)
	{
		if (!m_nodeSignalsDeactivated)
		{
			string nodeID = node.nodeDataModel()->caption().toStdString();
			if (p_manager->getNode(nodeID)->getNumOutputs() > 0)
			{
				ui->comboBoxPreviewNode->addItem(QString::fromStdString(nodeID));
			}
		}
	}

	void MainWindow::nodeDeleted(QtNodes::Node & node)
	{
		if (!m_nodeSignalsDeactivated)
		{
			QString nodeID = node.nodeDataModel()->caption();
			int index = ui->comboBoxPreviewNode->findText(nodeID);
			bool currentlyActive = ui->comboBoxPreviewNode->currentIndex() == index;
			ui->comboBoxPreviewNode->removeItem(index);
			if (currentlyActive)
			{
				ui->comboBoxPreviewNode->setCurrentIndex(0);
			}
			if (m_pParametersWidget && m_pParametersWidget->getNodeID() == nodeID)
			{
				hideParameters();
			}
		}
	}

	std::map<std::string, vec2i> MainWindow::computeNodePositions()
	{
		auto nodeIDs = p_manager->getNodeIDs();
		auto connections = p_manager->getNodeConnections();

		map <string, vector<string> > nodePredecessors;
		map <string, int> nodeLevel;

		for (auto node : nodeIDs)
		{
			nodeLevel[node] = -1;
			nodePredecessors[node].resize(0);
		}
		for (auto connection : connections)
		{
			string nodeFrom = get<0>(connection);
			string nodeTo = get<2>(connection);
			nodePredecessors[nodeTo].push_back(nodeFrom);
		}

		for (auto nodePredecessor : nodePredecessors)
		{
			if (nodePredecessor.second.size() == 0)
			{
				nodeLevel[nodePredecessor.first] = 0;
			}
		}
		for (size_t i = 0; i < nodeIDs.size(); i++)
		{
			for (auto& nodePredecessor : nodePredecessors)
			{
				if (nodePredecessor.second.size() != 0)
				{
					for (int pred = 0; pred < nodePredecessor.second.size(); pred++)
					{
						if (nodePredecessors[nodePredecessor.second[pred]].size() == 0)
						{
							nodeLevel[nodePredecessor.first] = max(nodeLevel[nodePredecessor.second[pred]] + 1, nodeLevel[nodePredecessor.first]);
							nodePredecessor.second.erase(nodePredecessor.second.begin() + pred);
							pred--;
						}
					}
				}
			}
		}

		int nodeLevels = -1;
		for (auto node : nodeIDs)
		{
			nodeLevels = max(nodeLevels, nodeLevel[node]);
		}
		nodeLevels++;

		vector<int> usedRowsPerLevel(nodeLevels, 0);
		map<string, vec2i> positionMap;
		for(auto node : nodeIDs)
		{
			int row = usedRowsPerLevel[nodeLevel[node]];
			usedRowsPerLevel[nodeLevel[node]]++;
			positionMap[node] = { nodeLevel[node], row };
		}

		return positionMap;
	}

	void MainWindow::setLogLevel()
	{
		QAction* pSender = dynamic_cast<QAction*>(sender());
		if (pSender)
		{
			logging::SeverityMask newLogLevel =
				(ui->actionLoglevelInfo->isChecked() ? logging::info : 0) |
				(ui->actionLoglevelLog->isChecked() ? logging::log : 0) |
				(ui->actionLoglevelWarning->isChecked() ? logging::warning : 0) |
				(ui->actionLoglevelError->isChecked() ? logging::error : 0) |
				(ui->actionLoglevelParameter->isChecked() ? logging::param : 0) |
				(ui->actionLoglevelExternals->isChecked() ? logging::external : 0);
			logging::Base::setLogLevel(newLogLevel);
		}
	}

	void MainWindow::setPreviewSize()
	{
		QAction* pSender = dynamic_cast<QAction*>(sender());
		if (pSender)
		{
			//uncheck the others
			ui->actionPreviewSizeSmall->setChecked(false);
			ui->actionPreviewSizeMedium->setChecked(false);
			ui->actionPreviewSizeLarge->setChecked(false);
			ui->actionPreviewSizeWambo->setChecked(false);

			QString newPreviewSize = pSender->text();

			//Set new preview size
			if (newPreviewSize == ui->actionPreviewSizeSmall->text())
			{
				m_previewSize = QSize(100, 100);
				ui->actionPreviewSizeSmall->setChecked(true);
			}
			else if (newPreviewSize == ui->actionPreviewSizeMedium->text())
			{
				m_previewSize = QSize(350, 400);
				ui->actionPreviewSizeMedium->setChecked(true);
			}
			else if (newPreviewSize == ui->actionPreviewSizeLarge->text())
			{
				m_previewSize = QSize(700, 800);
				ui->actionPreviewSizeLarge->setChecked(true);
			}
			else if (newPreviewSize == ui->actionPreviewSizeWambo->text())
			{
				m_previewSize = QSize(1200, 1200);
				ui->actionPreviewSizeWambo->setChecked(true);
			}

			//set the new preview size
			if (m_preview)
			{
				m_preview->setPreviewSize(m_previewSize);
			}
		}
	}

	void MainWindow::setLinearInterpolation()
	{
		bool linearInterpolation = ui->actionPreviewLinearInterpolation->isChecked();

		if (m_preview)
		{
			m_preview->setLinearInterpolation(linearInterpolation);
		}
	}

	void MainWindow::startNodes()
	{
		ui->pushButtonStart->setDisabled(true);

		SupraManager::Get()->setQuitCallback(std::bind(&MainWindow::quitCallback, this));

		SupraManager::Get()->startOutputs();
		SupraManager::Get()->startInputs();

		ui->pushButtonStop->setEnabled(true);
		ui->pushButtonSequence->setEnabled(true);
		ui->pushButtonFreeze->setEnabled(true);
		ui->pushButtonResetFreeze->setEnabled(true);
		m_started = true;
	}

	void MainWindow::stopNodes()
	{
		m_started = false;
		ui->pushButtonStop->setDisabled(true);
		ui->pushButtonSequence->setDisabled(true);
		ui->pushButtonFreeze->setDisabled(true);
		ui->pushButtonResetFreeze->setDisabled(true);
		//stop inputs
		SupraManager::Get()->stopAndWaitInputs();

		//wait for remaining messages to be processed
		SupraManager::Get()->waitForGraph();
	}

	void MainWindow::startSequence()
	{
		if (m_started)
		{
			if (!m_sequenceStarted)
			{
				ui->pushButtonSequence->setText("Sequence Stop");
				SupraManager::Get()->startOutputsSequence();
				m_sequenceStarted = true;
			}
		}
	}

	void MainWindow::stopSequence()
	{
		if (m_started)
		{
			if (m_sequenceStarted)
			{
				ui->pushButtonSequence->setText("Sequence Start");
				SupraManager::Get()->stopOutputsSequence();
				m_sequenceStarted = false;
			}
		}
	}

	void MainWindow::startStopSequence()
	{
		if (m_started)
		{
			if (!m_sequenceStarted)
			{
				startSequence();
			}
			else
			{
				stopSequence();
			}
		}
	}

	void MainWindow::updateNodeTimings()
	{
		m_pNodeScene->iterateOverNodeData(
			[this](QtNodes::NodeDataModel* nodeDataModel) {
			NodeExplorerDataModel* specific = dynamic_cast<NodeExplorerDataModel*>(nodeDataModel);
			if (specific)
			{
				string nodeID = nodeDataModel->caption().toStdString();
				auto node = p_manager->getNode(nodeID);

				string timingInfo = node->getTimingInfo();
				string newText;
				if (timingInfo != "")
				{
					newText += "(" + timingInfo + ")";
				}
				specific->setTimingText(newText);
			}
		});
	}

	void MainWindow::updateFreezeTimer()
	{
		// check the current mouse position
		QPoint currentCursorPosition = QCursor::pos();
		if (currentCursorPosition != m_previousCursorPosition)
		{
			// The mouse has been moved, we assume that we can reset the freeze timer
			p_manager->resetFreezeTimeout();
			m_previousCursorPosition = currentCursorPosition;
		}

		if (p_manager->inputsFrozen())
		{ 
			ui->pushButtonFreeze->setText(QString("Unfreeze"));
		}
		else
		{
			int32_t timeout = p_manager->getFreezeTimeout();
			int32_t timeoutMin = timeout / 60;
			int32_t timeoutSec = timeout % 60;
			ui->pushButtonFreeze->setText(QString("Freeze %1:%2").arg(timeoutMin).arg(timeoutSec, 2, 10, QChar('0')));
		}
	}

	void MainWindow::toogleFreeze()
	{
		if (p_manager->inputsFrozen())
		{
			p_manager->unfreezeInputs();
		}
		else
		{
			p_manager->freezeInputs();
		}
		updateFreezeTimer();
	}

	void MainWindow::resetFreezeTimer()
	{
		p_manager->resetFreezeTimeout();
		if (p_manager->inputsFrozen())
		{
			p_manager->unfreezeInputs();
		}
		updateFreezeTimer();
	}

	void MainWindow::previewSelected(const QString & text)
	{
		if (m_preview)
		{
			string existingNodeID = m_preview->getNodeID();
			string existingSrcID = existingNodeID.substr(5, existingNodeID.length() - 5 - 2);

			p_manager->disconnect(existingSrcID, 0, existingNodeID, 0);
			p_manager->removeNode(existingNodeID);

			m_preview->removePreviews();
			m_preview->setParent(nullptr);

			m_preview = nullptr;
		}

		string nodeID = text.toStdString();
		if (nodeID != "")
		{
			string previewNodeID = "PREV_" + nodeID + "_" + stringify(0);
			p_manager->addNodeConstruct<previewBuilderQT>(
				previewNodeID, "Preview " + nodeID + stringify(0), "previewBuilderQT", 
				ui->group_previews, ui->verticalLayoutPreviews, m_previewSize, m_previewLinearInterpolation);
			p_manager->connect(nodeID, 0, previewNodeID, 0);

			previewBuilderQT* pPreview = dynamic_cast<previewBuilderQT*>(p_manager->getNode(previewNodeID).get());
			if (pPreview)
			{
				m_preview = pPreview;
			}
		}
	}
}
