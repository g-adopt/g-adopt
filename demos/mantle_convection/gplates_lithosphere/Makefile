demos/mantle_convection/gplates_lithosphere/Makefile include ../../../rules.mk

.PHONY: all clean check data plate_data thickness_data craton_data

all: lithosphere_output.pvd

lithosphere_output.pvd : category := mantle_convection
lithosphere_output.pvd : ncpus := 4
lithosphere_output.pvd: gplates_lithosphere.py data
	$(run-python)

data: plate_data thickness_data craton_data

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

craton_files := Craton_Boundaries_Inferred.shp Craton_Boundaries_Inferred.shx Craton_Boundaries_Inferred.dbf Craton_Boundaries_Inferred.prj
craton_data: $(craton_files)

$(craton_files):
	@echo "Downloading craton boundary shapefiles..."
	curl -L -o Craton_Boundaries_Inferred.shp "https://raw.githubusercontent.com/EarthByte/Craton_Boundaries/main/Craton_Data/Craton_Boundaries_Inferred.shp"
	curl -L -o Craton_Boundaries_Inferred.shx "https://raw.githubusercontent.com/EarthByte/Craton_Boundaries/main/Craton_Data/Craton_Boundaries_Inferred.shx"
	curl -L -o Craton_Boundaries_Inferred.dbf "https://raw.githubusercontent.com/EarthByte/Craton_Boundaries/main/Craton_Data/Craton_Boundaries_Inferred.dbf"
	curl -L -o Craton_Boundaries_Inferred.prj "https://raw.githubusercontent.com/EarthByte/Craton_Boundaries/main/Craton_Data/Craton_Boundaries_Inferred.prj"

clean:
	rm -f *.pvd *.h5
	rm -rf lithosphere_output

check: lithosphere_output.pvd
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
