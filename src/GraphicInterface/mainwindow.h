#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <atomic>

#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QShortcut>
#include <QSize>
#include <QUuid>

#include <nodes/FlowScene>
#include <nodes/FlowView>
#include <nodes/Node>

#include <vec.h>

namespace Ui {
	class MainWindow;
}
namespace supra
{
	class SupraManager;
	class previewBuilderQT;
	class parametersWidget;

	/// The main window of the SUPRA GUI.
	/// Allows to change parameters during runtime and to preview the data streams
	class MainWindow : public QMainWindow
	{
		Q_OBJECT

	public:
		/// Default constructor
		explicit MainWindow(QWidget *parent = 0);
		~MainWindow();

		/// Is used to perform all steps necessary for closing the gui
		void prepareForClose();
		/// A callback that is registered with the SupraManager, in case any other 
		/// component requires the application to stop
		void quitCallback();
		/// On close of the window, the inputs are stopped and 
		/// this function waits for all nodes to finish
		void closeEvent(QCloseEvent *event);

		/// Loads the given configuration file,
		/// i.e. creates the graph as specified in the config
		void loadConfigFile(const QString & filename);

		/// Tries to catch as many key presses within the main window as possible.
		/// This serves as a trigger to reset the freeze timer, as input suggests 
		/// active use.
		void keyPressEvent(QKeyEvent* keyEvent);
	private:
		/// Slot that is called on selection in the node list.
		/// Shows a parameter widget for the currently selected node
		void showParametersFromList(QString nodeID);
		void nodeSceneSelectionChanged();
		void connectionCreated(QtNodes::Connection& connection);
		void connectionDeleted(QtNodes::Connection& connection);
		void nodeCreated(QtNodes::Node& node);
		void nodeDeleted(QtNodes::Node& node);

		std::map<std::string, vec2i> computeNodePositions();

		Ui::MainWindow *ui;

		QShortcut* m_pSequenceShortcut;

		std::shared_ptr<SupraManager> p_manager;
		std::atomic_bool m_nodeSignalsDeactivated;
		bool m_started;
		bool m_sequenceStarted;
		QSize m_previewSize;
		bool m_previewLinearInterpolation;
		previewBuilderQT* m_preview;
		parametersWidget* m_pParametersWidget;
		QtNodes::FlowScene* m_pNodeScene;
		QtNodes::FlowView* m_pNodeView;
		std::map<std::string, QUuid> m_NodeSceneNodeIDs;
		std::map<QUuid, QtNodes::Node*> m_NodeSceneNodes;

		QPoint m_previousCursorPosition;

	public slots:
		/// Slot that asks the user for the path to a configuration file
		/// and subsequently loads it, i.e. creates the graph as specified in the config
		void loadConfigFileAction();
		/// Selects the log-level to be shown in the console
		void setLogLevel();
		/// Sets the size of the preview image widget
		void setPreviewSize();
		/// Slot that sets whether linear interpolation is used to rescale preview images
		void setLinearInterpolation();
		/// Starts all output-nodes and input-nodes
		void startNodes();
		/// Stops all input nodes, waits for them to finish and then waits 
		/// for the compute graph to finish processing
		void stopNodes();
		/// Slot that starts a sequence for all nodes
		/// A sequence can be for example one recording
		void startSequence();
		/// Slot that stops a sequence for all nodes
		void stopSequence();
		/// Slot that starts and stops a sequence for all nodes
		void startStopSequence();
		/// Slot that updates the node list to show the node call frequencies and runtimes
		void updateNodeTimings();
		/// Slot that update the freeze timer
		void updateFreezeTimer();
		/// Slot that freezes and unfreezes the acquisition
		void toogleFreeze();
		/// Slot to reset the freeze timer
		void resetFreezeTimer();
		/// Slot that exchanges the current preview widget (if any) 
		///  with a new one for the given node ID
		void previewSelected(const QString & text);

	signals:
		void externClose();
	};
}

#endif // MAINWINDOW_H
