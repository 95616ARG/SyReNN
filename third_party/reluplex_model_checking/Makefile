ROOT_DIR = check_properties
PROJECT_DIR = .
GLPK_DIR = $(PROJECT_DIR)/glpk-4.60

SUBDIRS += \

LOCAL_INCLUDES += \
	$(PROJECT_DIR)/common \
	$(GLPK_DIR)/installed/include \
	$(PROJECT_DIR)/reluplex \
	$(PROJECT_DIR)/nnet \

LINK_FLAGS += \
	-L$(GLPK_DIR)/installed/lib

LOCAL_LIBRARIES += \
	glpk

SOURCES += \
	nnet.cpp \
	bmc.cpp

TARGET = bmc.elf

include $(ROOT_DIR)/Rules.mk

vpath %.cpp $(PROJECT_DIR)/nnet
