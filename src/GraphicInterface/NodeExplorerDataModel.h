// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __NODEEXPLORERDATAMODEL_H__
#define __NODEEXPLORERDATAMODEL_H__

#include <nodes/NodeDataModel>
#include <QCheckBox>

namespace supra
{
	class NodeExplorerDataModel : public QtNodes::NodeDataModel
	{
		Q_OBJECT

	public:
		NodeExplorerDataModel(std::string nodeID, std::string nodeType);

		virtual QString caption() const;
		virtual QString name() const;
		virtual std::unique_ptr<NodeDataModel> clone() const;
		virtual unsigned int nPorts(QtNodes::PortType portType) const;
		virtual QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const;
		virtual void setInData(std::shared_ptr<QtNodes::NodeData> nodeData, QtNodes::PortIndex port);
		virtual std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex port);
		virtual QWidget * embeddedWidget();
		QCheckBox * embeddedCheckbox();
		void setTimingText(const std::string& text);

	private:
		std::string m_nodeID;
		std::string m_nodeType;

		QCheckBox * m_labeledCheckBox;
	};
}

#endif //!__NODEEXPLORERDATAMODEL_H__
