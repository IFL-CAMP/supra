#include "mainwindow.h"
#include <locale>
#include <clocale>
#include <QApplication>
#include <QLocale>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <utilities/Logging.h>

using namespace supra;

int main(int argc, char *argv[])
{
	std::setlocale(LC_ALL, "C");
	std::locale::global(std::locale::classic());

	QLocale::setDefault(QLocale("C"));

	QApplication a(argc, argv);

	QCommandLineParser parser;
	parser.addOption(QCommandLineOption(QStringList() << "config" << "c", "xml config file", "configFile"));
	parser.addOption(QCommandLineOption(QStringList() << "autostart" << "a", "Autostart pipeline"));
	parser.process(a);

	bool autostart = parser.isSet("autostart");
	QString configFile = "";
	bool configFileSet = parser.isSet("config");
	if (configFileSet)
	{
		configFile = parser.value("config");
	}
	else if (autostart)
	{
		logging::log_error("Error: Autostart can only be used when specifing a configFile.");
		return 1;
	}

	MainWindow w;
    w.show();

	if (configFileSet)
	{
		w.loadConfigFile(configFile);
		if (autostart)
		{
			w.startNodes();
		}
	}

    return a.exec();
}
