include ../../../rules.mk

.PHONY: all clean check data

all: lithosphere_output.pvd

lithosphere_output.pvd : category := mantle_convection
lithosphere_output.pvd: gplates_lithosphere.py data
	$(run-python)

data:
	@echo "Downloading plate reconstruction files..."
	@if [ ! -d "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2" ]; then \
		curl -L -o muller2022.zip "https://earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2022_SE/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.zip" && \
		unzip -q muller2022.zip && \
		rm muller2022.zip; \
	fi

clean:
	rm -f *.pvd *.h5
	rm -rf lithosphere_output

check: lithosphere_output.pvd
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
