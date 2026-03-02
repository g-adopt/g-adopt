include ../../../rules.mk

.PHONY: all clean check data plate_data thickness_data

all: initial_condition_output.pvd

initial_condition_output.pvd : category := mantle_convection
initial_condition_output.pvd : ncpus := 4
initial_condition_output.pvd: gplates_initial_condition.py data
	$(run-python)

data: plate_data thickness_data

plate_data:
	@echo "Downloading plate reconstruction files..."
	@if [ ! -d "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2" ]; then \
		curl -L -o muller2022.zip "https://earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2022_SE/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.zip" && \
		unzip -q muller2022.zip && \
		rm muller2022.zip; \
	fi

thickness_data:
	@echo "Downloading continental thickness data..."
	@if [ ! -f "lithospheric_thickness_mesh.h5" ]; then \
		curl -L -o lithospheric_thickness_mesh.h5 "https://data.gadopt.org/demos/lithospheric_thickness_mesh.h5"; \
	fi

clean:
	rm -f *.pvd *.h5
	rm -rf initial_condition_output

check: initial_condition_output.pvd
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
