#include "parametersWidget.h"
#include "ui_parametersWidget.h"

#include <memory>
#include <QFileDialog>
#include <QString>
#include <QScrollBar>
#include <QLabel>

#include <SupraManager.h>
#include <utilities/Logging.h>
#include "parameterWidget.h"

namespace supra
{
	using namespace std;

	parametersWidget::parametersWidget(const QString & nodeID, QWidget *parent) :
		QWidget(parent),
		ui(new Ui::parametersWidget)
	{
		ui->setupUi(this);

		//Fetch node pointer from manager
		p_node = SupraManager::Get()->getNode(nodeID.toStdString());
		logging::log_error_if(!p_node, "GUI: Error fetching node '", nodeID.toStdString(), ",");

		auto parameterNames = p_node->getValueRangeDictionary()->getKeys();

		//add all parameter elements
		for (size_t i = 0; i < parameterNames.size(); i++)
		{
			auto paramWidget = new parameterWidget(nodeID, QString::fromStdString(parameterNames[i]), this);
			m_parameterWidgets.push_back(paramWidget);
			paramWidget->setupUi(paramWidget);
			ui->formLayout->addRow(paramWidget->getDisplayName(), paramWidget);
		}
	}

	parametersWidget::~parametersWidget()
	{
		delete ui;
	}
	QString parametersWidget::getNodeID()
	{
		return QString::fromStdString(p_node->getNodeID());
	}
}
