#ifndef __PARAMETERWIDGET_H__
#define __PARAMETERWIDGET_H__

#include <memory>
#include <string>

#include <QWidget>
#include <QVBoxLayout>
#include <QAbstractSpinBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>

#include <SupraManager.h>
#include <utilities/Logging.h>
#include <utilities/utility.h>

#include <cmath>
#include <type_traits>

namespace supra
{
	class parameterWidget;

	/// Interface to the templated parameterWidgets
	class parameterWidgetTypeHandlerGeneral
	{
	public:
		/// Adds the QT elements for this parameter to the given layout
		virtual void addElements(QVBoxLayout* targetLayout) = 0;
		/// Removes the QT elements this parameterHandles has created
		virtual void removeElements() = 0;
		/// Returns the display name of the parameter to be edited
		virtual const std::string & getDisplayName() = 0;
		/// Function to be called when the value of the widgets has changed
		virtual void valueChanged(const std::string & text) = 0;
		/// Function to be called when the value of the widgets has changed
		virtual void valueChanged(const Qt::CheckState & state) = 0;
		virtual ~parameterWidgetTypeHandlerGeneral() {}
	};

	/// Templated parameter widget: Handles numeric types
	template<typename ValueType>
	class parameterWidgetTypeHandler : public parameterWidgetTypeHandlerGeneral
	{
	public:
		/// Constructor templated parameter widget: Handles numeric types
		/// takes the ID of the node and the name of the parameter to display
		/// as well as the parent widget to attach the elements to
		parameterWidgetTypeHandler(const std::string & nodeID, const std::string & parameterName, QWidget* widget)
			: p_spinBox(nullptr)
			, p_comboBox(nullptr)
			, m_eventOverride(false)
		{
			p_node = SupraManager::Get()->getNode(nodeID);
			p_range = p_node->getValueRangeDictionary()->get<ValueType>(parameterName);
			m_paramName = parameterName;
			p_widget = widget;
		}

		//Templated to deal with different target types.
		// general implementation for scalar types
		virtual void addElements(QVBoxLayout* targetLayout)
		{
			static_assert(std::is_arithmetic<ValueType>::value ||
				std::is_enum<ValueType>::value, 
				"Missing addElements-implementation for type");
			if (p_range->isUnrestricted())
			{
				//Add Spinbox
				if (std::is_integral<ValueType>::value)
				{
					auto pBox = new QSpinBox(p_widget);
					p_spinBox = pBox;
					using T = typename std::common_type<ValueType, int>::type;
					pBox->setMinimum(max((T)(std::numeric_limits<ValueType>::min()), (T)(std::numeric_limits<int>::min())));
					pBox->setMaximum(min((T)(std::numeric_limits<ValueType>::max()), (T)(std::numeric_limits<int>::max())));
					pBox->setSingleStep(1.0);
				}
				else {
					auto pBox = new QDoubleSpinBox(p_widget);
					p_spinBox = pBox;
					pBox->setMinimum(std::numeric_limits<ValueType>::min());
					pBox->setMaximum(std::numeric_limits<ValueType>::max());
					pBox->setSingleStep(1.0);
				}
				p_spinBox->setObjectName(QString::fromStdString("spinBox_" + m_paramName));
				//p_spinBox->setText(QString::fromStdString(p_range->getDisplayName()));

				targetLayout->addWidget(p_spinBox);
				p_widget->connect(p_spinBox, SIGNAL(valueChanged(const QString &)),
					p_widget, SLOT(valueChanged(const QString &)));
			}
			else if (p_range->isContinuous())
			{
				//Add Slider + Spinbox
				if (std::is_integral<ValueType>::value)
				{
					auto pBox = new QSpinBox(p_widget);
					p_spinBox = pBox;
					pBox->setMinimum(p_range->getContinuous().first);
					pBox->setMaximum(p_range->getContinuous().second);
				}
				else {
					auto pBox = new QDoubleSpinBox(p_widget);
					p_spinBox = pBox;
					pBox->setMinimum(p_range->getContinuous().first);
					pBox->setMaximum(p_range->getContinuous().second);
					double singleStep = std::pow(10, std::floor(std::log10(
						(p_range->getContinuous().second - p_range->getContinuous().first) / 100)));
					pBox->setSingleStep(singleStep);
				}
				p_spinBox->setObjectName(QString::fromStdString("spinBox_" + m_paramName));
				//p_spinBox->setText(QString::fromStdString(p_range->getDisplayName()));

				targetLayout->addWidget(p_spinBox);
				p_widget->connect(p_spinBox, SIGNAL(valueChanged(const QString &)),
					p_widget, SLOT(valueChanged(const QString &)));
			}
			else //Discrete
			{
				//Add Dropdownbox
				p_comboBox = new QComboBox(p_widget);
				for (auto value : p_range->getDiscrete())
				{
					p_comboBox->addItem(QString::fromStdString(stringify(value)));
				}
				p_comboBox->setObjectName(QString::fromStdString("comboBox_" + m_paramName));
				targetLayout->addWidget(p_comboBox);

				p_widget->connect(p_comboBox, SIGNAL(currentTextChanged(const QString)),
					p_widget, SLOT(valueChanged(const QString &)));
			}
			setCurrentValue();
		}
		virtual void removeElements()
		{
			if (p_spinBox)
				delete p_spinBox;
			if (p_comboBox)
				delete p_comboBox;
		}

		virtual const std::string & getDisplayName()
		{
			return p_range->getDisplayName();
		}

		virtual void valueChanged(const std::string & text)
		{
			if (!m_eventOverride)
			{
				ValueType v = from_string<ValueType>(text);
				p_node->changeConfig(m_paramName, v);

				setCurrentValue();
			}
		}
		virtual void valueChanged(const Qt::CheckState & state) {};

		/// Sets the QT elements to the current parameter value.
		void setCurrentValue()
		{
			m_eventOverride = true;
			ValueType v = p_node->getConfigurationDictionary()->get<ValueType>(m_paramName);
			if (p_spinBox)
			{
				auto normalSpinBox = dynamic_cast<QSpinBox*>(p_spinBox);
				if (normalSpinBox)
				{
					normalSpinBox->setValue(v);
				}
				auto doubleSpinBox = dynamic_cast<QDoubleSpinBox*>(p_spinBox);
				if (doubleSpinBox)
				{
					doubleSpinBox->setValue(v);
				}
			}
			if (p_comboBox)
			{
				int nextIndex = p_comboBox->findText(QString::fromStdString(stringify(v)));
				p_comboBox->setCurrentIndex(nextIndex);
			}
			m_eventOverride = false;
		}
	private:
		std::shared_ptr<AbstractNode> p_node;
		const ValueRangeEntry<ValueType> * p_range;
		std::string m_paramName;
		QWidget* p_widget;
		QAbstractSpinBox* p_spinBox;
		QComboBox* p_comboBox;
		bool m_eventOverride;
	};

	/// Specialization of the parameterWidget to handle strings
	template<>
	class parameterWidgetTypeHandler<std::string> : public parameterWidgetTypeHandlerGeneral
	{
	public:
		/// Constructor of the speciaized parameterWidget
		parameterWidgetTypeHandler(const std::string & nodeID, const std::string & parameterName, QWidget* widget)
			: p_comboBox(nullptr)
			, p_textBox(nullptr)
			, m_eventOverride(false)
		{
			p_node = SupraManager::Get()->getNode(nodeID);
			p_range = p_node->getValueRangeDictionary()->get<std::string>(parameterName);
			m_paramName = parameterName;
			p_widget = widget;
		}

		virtual void addElements(QVBoxLayout* targetLayout)
		{
			if (p_range->isUnrestricted())
			{
				//Add textbox
				p_textBox = new QLineEdit(p_widget);

				p_textBox->setObjectName(QString::fromStdString("textBoxBox_" + m_paramName));
				targetLayout->addWidget(p_textBox);
				p_widget->connect(p_textBox, SIGNAL(editingFinished()),
					p_widget, SLOT(lineEditingFinished()));
			}
			else if (p_range->isContinuous())
			{
				//This combination does not make sense for strings
				logging::log_error("Could not create parameterTypeHandler<std::string> for paramter '",
					m_paramName, "'. Continuous range is not possible for strings");
			}
			else //Discrete
			{
				//Add Dropdownbox
				p_comboBox = new QComboBox(p_widget);
				for (auto value : p_range->getDiscrete())
				{
					p_comboBox->addItem(QString::fromStdString(value));
				}
				p_comboBox->setObjectName(QString::fromStdString("comboBox_" + m_paramName));
				targetLayout->addWidget(p_comboBox);
				p_widget->connect(p_comboBox, SIGNAL(currentTextChanged(const QString &)),
					p_widget, SLOT(valueChanged(const QString &)));
			}
			setCurrentValue();
		}
		virtual void removeElements()
		{
			if (p_comboBox)
				delete p_comboBox;
			if (p_textBox)
				delete p_textBox;
		}
		virtual const std::string & getDisplayName()
		{
			return p_range->getDisplayName();
		}

		/// function to be called when the value of the gui elements has changed
		virtual void valueChanged(const std::string & text)
		{
			if (!m_eventOverride)
			{
				p_node->changeConfig<std::string>(m_paramName, text);

				setCurrentValue();
			}
		}
		virtual void valueChanged(const Qt::CheckState & state) {};

		/// Sets the QT elements to the current parameter value.
		void setCurrentValue()
		{
			m_eventOverride = true;
			std::string v = p_node->getConfigurationDictionary()->get<std::string>(m_paramName);
			if (p_comboBox)
			{
				int nextIndex = p_comboBox->findText(QString::fromStdString(v));
				p_comboBox->setCurrentIndex(nextIndex);
			}
			if (p_textBox)
			{
				p_textBox->setText(QString::fromStdString(v));
			}
			m_eventOverride = false;
		}
	private:
		std::shared_ptr<AbstractNode> p_node;
		const ValueRangeEntry<std::string>* p_range;
		std::string m_paramName;
		QWidget* p_widget;
		QComboBox* p_comboBox;
		QLineEdit* p_textBox;
		bool m_eventOverride;
	};

	/// Specialization of the parameterWidget to handle boolean parameters
	template<>
	class parameterWidgetTypeHandler<bool> : public parameterWidgetTypeHandlerGeneral
	{
	public:
		/// Constructor for the bool-speciaized parameterWidgetTypeHandler
		parameterWidgetTypeHandler(const std::string & nodeID, const std::string & parameterName, QWidget* widget)
			: p_checkbox(nullptr)
			, m_eventOverride(false)
		{
			p_node = SupraManager::Get()->getNode(nodeID);
			p_range = p_node->getValueRangeDictionary()->get<bool>(parameterName);
			m_paramName = parameterName;
			p_widget = widget;
		}

		//Templated to deal with different target types.
		// general implementation for scalar types
		virtual void addElements(QVBoxLayout* targetLayout)
		{
			//Add CheckBox
			p_checkbox = new QCheckBox(p_widget);

			p_checkbox->setObjectName(QString::fromStdString("checkBox_" + m_paramName));
			targetLayout->addWidget(p_checkbox);
			p_widget->connect(p_checkbox, SIGNAL(stateChanged(int)),
				p_widget, SLOT(valueChanged(int)));

			setCurrentValue();
		}
		virtual void removeElements()
		{
			if (p_checkbox)
				delete p_checkbox;
		}
		virtual const std::string & getDisplayName()
		{
			return p_range->getDisplayName();
		}
		/// function to be called when the value of the gui elements has changed
		virtual void valueChanged(const std::string & text) {};
		virtual void valueChanged(const Qt::CheckState & state)
		{
			if (!m_eventOverride)
			{
				p_node->changeConfig<bool>(m_paramName, state == Qt::Checked);

				setCurrentValue();
			}
		}

		/// Sets the QT elements to the current parameter value.
		void setCurrentValue()
		{
			m_eventOverride = true;
			bool v = p_node->getConfigurationDictionary()->get<bool>(m_paramName);
			p_checkbox->setCheckState((v ? Qt::Checked : Qt::Unchecked));
			m_eventOverride = false;
		}
	private:
		std::shared_ptr<AbstractNode> p_node;
		const ValueRangeEntry<bool>* p_range;
		std::string m_paramName;
		QWidget* p_widget;
		QCheckBox* p_checkbox;
		bool m_eventOverride;
	};

	/**
	 * A widget to inspect and manipulate one parameter of a node
	 * 
	 * parameterWidget shows one parameter of a node and allows to inspect and 
	 * modifiy its value, as specified by the \see ValueRangeDictionary.
	 * To achieve that, it has an instance of the templated class \see parameterWidgetTypeHandler,
	 * to serve as an input to the configuration.
	 */
	class parameterWidget : public QWidget
	{
		Q_OBJECT

	public:
		/// Constructor of parameterWidget, takes the ID of node and the name of the parameter
		/// that is to be displayed.
		explicit parameterWidget(const QString & nodeID, const QString& parameterName, QWidget *parent);

		~parameterWidget();

		/// Sets all QT properties of this widget and creates the elements of the typeHandler
		void setupUi(QWidget *parametersWidget);
		/// Returns the displayname of the parameter of interest
		QString getDisplayName();
	private:
		QString m_nodeID;
		QString m_paramName;
		std::unique_ptr<parameterWidgetTypeHandlerGeneral> m_pTypeHandler;
		QVBoxLayout *m_verticalLayout;

	public slots:
		/// Slots for the parameter changes of the various widget types
		void valueChanged(const QString & text);
		/// Slots for the parameter changes of the various widget types
		void valueChanged(int state);
		/// Slots for the parameter changes of the various widget types
		void lineEditingFinished();

	private:
		friend parameterWidgetTypeHandlerGeneral;

	};
}
#endif //!__PARAMETERWIDGET_H__
