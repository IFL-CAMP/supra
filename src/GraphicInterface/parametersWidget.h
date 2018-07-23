#ifndef __PARAMETERSWIDGET_H__
#define __PARAMETERSWIDGET_H__

#include <memory>

#include <QWidget>

namespace Ui {
	class parametersWidget;
}

namespace supra
{
	class RecordObject;
	class AbstractNode;
	class parameterWidget;

	/**
	 * A widget to show all parameters of a node.
	 *
	 * parametersWidget shows all parameters of a node and allows the user to change
	 * them. For that it creates a \see parameterWidget for each parameter.
	 */
	class parametersWidget : public QWidget
	{
		Q_OBJECT

	public:
		/// Constructor of parametersWidget. It takes the ID of the node to display.
		explicit parametersWidget(const QString & nodeID, QWidget *parent = 0);
		~parametersWidget();
		QString getNodeID();

	private:
		Ui::parametersWidget *ui;

		std::shared_ptr<AbstractNode> p_node;

		std::vector<parameterWidget* > m_parameterWidgets;
	};
}

#endif //!__PARAMETERSWIDGET_H__
