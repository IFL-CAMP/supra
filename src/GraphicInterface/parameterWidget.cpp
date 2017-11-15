
#include <memory>
#include <string>
#include <utilities/Logging.h>
#include <SupraManager.h>

#include "parameterWidget.h"

namespace supra
{
	using namespace ::std;

	parameterWidget::parameterWidget(const QString & nodeID, const QString& parameterName, QWidget *parent)
		: m_verticalLayout(nullptr)
	{
		m_nodeID = nodeID;
		m_paramName = parameterName;
		string paramName = m_paramName.toStdString();
		//create the TypeHandler
		auto p_node = SupraManager::Get()->getNode(nodeID.toStdString());
		switch (p_node->getValueRangeDictionary()->getType(paramName))
		{
		case TypeBool:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<bool>(nodeID.toStdString(), paramName, this));
			break;
		case TypeInt8:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<int8_t>(nodeID.toStdString(), paramName, this));
			break;
		case TypeUint8:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<uint8_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeInt16:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<int16_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeUint16:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<uint16_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeInt32:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<int32_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeUint32:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<uint32_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeInt64:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<uint64_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeUint64:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<uint64_t>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeFloat:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<float>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeDouble:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<double>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeString:
			m_pTypeHandler = unique_ptr<parameterWidgetTypeHandlerGeneral>(
				new parameterWidgetTypeHandler<string>(nodeID.toStdString(), paramName, this));
			break;
			break;
		case TypeValueUnknown:
		default:
			logging::log_error("Could not create parameter widget for '", nodeID.toStdString(), "', '",
				paramName, "'");
			break;
		}
	}

	parameterWidget::~parameterWidget()
	{
		m_pTypeHandler->removeElements();
	}

	void parameterWidget::setupUi(QWidget *parametersWidget)
	{
		if (parametersWidget->objectName().isEmpty())
			parametersWidget->setObjectName(QStringLiteral("parameterWidget"));
		parametersWidget->resize(100, 35);
		QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
		sizePolicy.setHorizontalStretch(0);
		sizePolicy.setVerticalStretch(0);
		sizePolicy.setHeightForWidth(parametersWidget->sizePolicy().hasHeightForWidth());
		parametersWidget->setSizePolicy(sizePolicy);
		m_verticalLayout = new QVBoxLayout(parametersWidget);
		m_verticalLayout->setSpacing(0);
		m_verticalLayout->setContentsMargins(0, 0, 0, 0);
		m_verticalLayout->setObjectName(QStringLiteral("verticalLayout"));

		m_pTypeHandler->addElements(m_verticalLayout);

		//radioButton = new QRadioButton(parametersWidget);
		//radioButton->setObjectName(QStringLiteral("radioButton"));
		//radioButton->setText(QApplication::translate("parametersWidget", "RadioButton", 0));

		//verticalLayout->addWidget(radioButton);
	} // setupUi

	QString parameterWidget::getDisplayName()
	{
		return QString::fromStdString(m_pTypeHandler->getDisplayName());
	}

	void parameterWidget::valueChanged(const QString & text)
	{
		m_pTypeHandler->valueChanged(text.toStdString());
	}

	void parameterWidget::valueChanged(int state)
	{
		m_pTypeHandler->valueChanged((Qt::CheckState)state);
	}

	void parameterWidget::lineEditingFinished()
	{
		QLineEdit* lineEdit = dynamic_cast<QLineEdit*>(sender());
		if (lineEdit)
		{
			m_pTypeHandler->valueChanged(lineEdit->text().toStdString());
		}
	}
}
