#****************************************************************************
# Makefile
# openacousticdevices.info
# November 2020
#****************************************************************************

# Set the toolpath parameter to the location of the GNU Arm Embedded Toolchain

TOOLCHAIN_PATH = --- INSERT PATH TO GNU ARM EMBEDDED TOOLCHAIN HERE (e.g. /Library/Toolchains/) ---

TOOLCHAIN_VERSION = gcc-arm-none-eabi-10.3-2021.10

# Determine the path to the toolchain binary

# Conformer latency firmware: the toolchain on the PATH
TOOLPATH =

# These are the locations of the source and header files

# Conformer latency firmware: the GPS library (gps.c), CMSIS 6 core, CMSIS-DSP (the front end's FFT), CMSIS-NN (the Conformer's kernels), the exported model
INC = ../CMSIS_6/CMSIS/Core/Include ../device/inc ../emlib/inc ../emusb/inc ../drivers/inc ../fatfs/inc ../gps/inc ../inc ../CMSIS-DSP/Include ../CMSIS-DSP/PrivateInclude ../CMSIS-DSP/Source/TransformFunctions ../CMSIS-DSP/Source/CommonTables ../CMSIS-DSP/Source/FastMathFunctions ../CMSIS-NN/Include ../model
SRC = ../device/src ../emlib/src ../emusb/src ../drivers/src ../fatfs/src ../gps/src ../src ../model ../CMSIS-NN/Source/FullyConnectedFunctions ../CMSIS-NN/Source/NNSupportFunctions ../CMSIS-NN/Source/SoftmaxFunctions

# Set the name of the output files

FILENAME = audiomoth

# This is the location of the resulting object files

OBJPATH = ./objects/

# Set the target EFM32WG to that used in AudioMoth

TARGET = EFM32WG380F256

# The following code generates the list of objects and the search path of source and header files

VPATH = $(SRC)

IFLAGS = $(foreach d, $(INC), -I$d)

_CSRC = $(notdir $(foreach d, $(SRC), $(wildcard $d/*.c)))
_SSRC = $(notdir $(foreach d, $(SRC), $(wildcard $d/*.s)))

_OBJ = $(_CSRC:.c=.o) $(_SSRC:.s=.o)

OBJ = $(sort $(foreach d, $(_OBJ), $(OBJPATH)$d))

DEP = $(OBJ:.o=.d)

# These are the compilation settings

CC = $(TOOLPATH)arm-none-eabi-gcc

CSIZE = $(TOOLPATH)arm-none-eabi-size

COBJCOPY = $(TOOLPATH)arm-none-eabi-objcopy

COBJDUMP = $(TOOLPATH)arm-none-eabi-objdump

CFLAGS = -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mthumb -Wall '-DARM_MATH_CM4=1' '-D$(TARGET)=1'

# Conformer latency firmware: CMSIS 6 renamed the trace port interface TPI (audiomoth.c) to TPIU
CFLAGS += '-DTPI=TPIU'

DFLAGS = -MMD

# Finally the build rules

$(OBJPATH)%.o: %.c
	@mkdir -p $(OBJPATH)
	@echo 'Building' $@
	@$(CC) $(CFLAGS) $(DFLAGS) -O3 -ffunction-sections -fdata-sections -std=c99 -c -o "$@" "$<" $(IFLAGS)

$(OBJPATH)%.o: %.s
	@mkdir -p $(OBJPATH)
	@echo 'Building' $@
	@$(CC) $(CFLAGS) $(DFLAGS) -x assembler-with-cpp -c -o "$@" "$<" $(IFLAGS)

$(FILENAME).bin: $(OBJ)
	@echo 'Building' $(FILENAME).axf
	@$(CC) $(CFLAGS) $(DFLAGS) -T "audiomoth.ld" -Xlinker --gc-sections -Xlinker -Map="$(FILENAME).map" --specs=nano.specs -o $(FILENAME).axf $(OBJ) -lm -Wl,--start-group -lgcc -lc -lnosys -Wl,--end-group
	@echo 'Building' $(FILENAME).lst
	@$(COBJDUMP) -d -t -S "$(FILENAME).axf" > "$(FILENAME).lst" 
	@echo 'Building' $(FILENAME).hex
	@$(COBJCOPY) -O ihex "$(FILENAME).axf" "$(FILENAME).hex"
	@echo 'Building' $(FILENAME).bin
	@$(COBJCOPY) -O binary "$(FILENAME).axf" "$(FILENAME).bin"
	@$(CSIZE) -A "$(FILENAME).axf"

-include $(DEP)

.PHONY: clean
clean:
	rm -f $(OBJPATH)*.o
	rm -f $(OBJPATH)*.d
	rm -f $(FILENAME).axf
	rm -f $(FILENAME).lst
	rm -f $(FILENAME).bin
	rm -f $(FILENAME).hex
	rm -f $(FILENAME).map
