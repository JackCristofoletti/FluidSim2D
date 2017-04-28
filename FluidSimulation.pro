#-------------------------------------------------
#
# Project created by QtCreator 2017-01-17T23:55:53
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FluidSimulation
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    src/Fluid/fluidquantity.cpp \
    src/Fluid/fluidsolver.cpp \
    src/Graphics/renderer.cpp \
    src/Gui/fluidglwidget.cpp

HEADERS  += mainwindow.h \
    src/Fluid/fluidquantity.h \
    src/Math/mathutil.h \
    src/Fluid/fluidsolver.h \
    src/Graphics/renderer.h \
    src/Gui/fluidglwidget.h

FORMS    += mainwindow.ui

DISTFILES += \
    shaders/quad.vert \
    shaders/quad.frag
