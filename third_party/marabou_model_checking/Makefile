# Based on
# https://github.com/NeuralNetworkVerification/Marabou/blob/cav_artifact/cpp_interface_example/Makefile
ROOT_DIR = ..

include $(ROOT_DIR)/Places.mk

SUBDIRS += \

LOCAL_INCLUDES += \
	.. \
	$(BASIS_FACTORIZATION_DIR) \
	$(CONFIGURATION_DIR) \
	$(ENGINE_DIR) \
	$(INPUT_PARSER_DIR) \

LINK_FLAGS += \

LOCAL_LIBRARIES += \

CFLAGS += \

include $(BASIS_FACTORIZATION_DIR)/Sources.mk
include $(COMMON_DIR)/Sources.mk
include $(COMMON_REAL_DIR)/Sources.mk
include $(CONFIGURATION_DIR)/Sources.mk
include $(ENGINE_DIR)/Sources.mk
include $(ENGINE_REAL_DIR)/Sources.mk
include $(INPUT_PARSER_DIR)/Sources.mk

vpath %.cpp $(BASIS_FACTORIZATION_DIR)
vpath %.cpp $(COMMON_DIR)
vpath %.cpp $(COMMON_REAL_DIR)
vpath %.cpp $(CONFIGURATION_DIR)
vpath %.cpp $(ENGINE_DIR)
vpath %.cpp $(ENGINE_REAL_DIR)
vpath %.cpp $(INPUT_PARSER_DIR)

SOURCES += \
	bmc.cpp \

TARGET = bmc.elf

include $(ROOT_DIR)/Rules.mk

#
# Local Variables:
# compile-command: "make -C . "
# tags-file-name: "../TAGS"
# c-basic-offset: 4
# End:
#
