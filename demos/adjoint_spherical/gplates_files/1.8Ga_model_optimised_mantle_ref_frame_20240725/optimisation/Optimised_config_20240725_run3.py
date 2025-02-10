import glob
import math
import os.path
import warnings


#########################################
# Optimisation configuration parameters #
#########################################


##########################################################################################
# Supported parallelisation methods (or None to disable parallelisation, eg, for testing).
#
MPI4PY = 0
IPYPARALLEL = 1

# Choose parallelisation method (or None to disable parallelisation, eg, for testing).
use_parallel = MPI4PY  # For example, to use with 'mpiexec -n <cores> python Optimised_APM.py'.
#use_parallel = None
##########################################################################################
# The root input data directory ('data/').
# This is the 'data/' sub-directory of the directory containing this source file.
datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', '')


# The data model to run the optimisation on.
# This should be the name of the sub-directory in 'data/'.
#data_model = 'Global_1000-0_Model_2017'
#data_model = 'Global_Model_WD_Internal_Release_2019_v2'
#data_model = 'Global_Model_WD_Internal_Release-EarthBytePlateMotionModel-TRUNK'
#data_model = 'SM2-Merdith_et_al_1_Ga_reconstruction_v1.1'
data_model = '1.8Ga_model_2024_07_25'


# The model name is suffixed to various output filenames.
if data_model.startswith('Global_Model_WD_Internal_Release'):
    model_name = "git_20230814_09f1e22_run1"
elif data_model == 'Global_1000-0_Model_2017':
    model_name = "git_20210802_ce53d67_run45"
elif data_model == 'Zahirovic_etal_2022_GDJ':
    model_name = "git_20230821_bb344f8_run2"
elif '1.8ga' in data_model.lower():
    model_name = "20240725_run3"
elif data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction'):
    model_name = "git_20231114_run1"
else:
    model_name = "run1"


# Start age.
if data_model == 'Zahirovic_etal_2022_GDJ':
    start_age = 410
elif '1.8ga' in data_model.lower():
    start_age = 1800
else:
    start_age = 1000

# End age.
#
# Note: You can set 'end_age' to a non-zero value if you are continuing an interrupted run.
#       In this case the workflow will attempt to re-use the existing partially optimised rotation file.
#       This can save a lot of time by skipping the optimisations already done by the interrupted optimisation run.
#       But be sure to set 'end_age' back to zero when finished.
#       Also this currently only works properly if 'interval' is the same for the interrupted and continued runs (as it should be).
end_age = 0

# Time interval.
if data_model.startswith('Global_Model_WD_Internal_Release'):
    interval = 5
elif data_model == 'Global_1000-0_Model_2017':
    # 5My works well with using a no-net-rotation reference frame (instead of PMAG), and is about as fast as 10My.
    interval = 5
else:
    interval = 5

# Seed model rotation poles are populated within a small circle of the search radius (degrees) about the reference pole.
search_radius = 180

# Number of seed models if the search radius were to cover the entire globe.
#
# In the past we typically used models=100 and search radius=60, which equates to model_density=400.
#
# NOTE: Currently, if 'search_type' is 'Uniform', then this will be rounded up to the nearest square (eg, 16, 25, 36).
model_density = 400

# Calculate the actual number of seed models used.
#
# Note: If 'search_radius' is changed then is automatically changed in proportion to the search area.
#       The proportion of globe covered by search is "0.5*(1-cos(search_radius))" where 1.0 means the entire globe.
#       This ensures the spatial density of seed models remains the same.
models = int(1e-4 + model_density * 0.5 * (1 - math.cos(math.radians(search_radius))))

# The original rotation files (relative to the 'data/' directory).
#
# Can either:
#   1) use glob to automatically find all the '.rot' files (you don't need to do anything), or
#   2) explicitly list all the rotation files (you need to list the filenames).
#
# 1) Automatically gather all '.rot' files (and make filenames relative to the 'data/' directory).
original_rotation_filenames = [os.path.relpath(abs_path, datadir) for abs_path in
        glob.glob(os.path.join(datadir, data_model, '*.rot'))]
# 2) Explicitly list all the input rotation files (must be relative to the 'data/' directory).
#original_rotation_filenames = [
#  data_model + '/rotation_file1.rot',
#  data_model + '/rotation_file2.rot',
#]

# The topology files (relative to the 'data/' directory).
#
# Can either:
#   1) use glob to automatically find all the '.gpml' files (you don't need to do anything), or
#   2) explicitly list all the topology files (you need to list the filenames).
#
# 1) Automatically gather all '.gpml' and '.gpmlz' files (and make filenames relative to the 'data/' directory).
#topology_filenames = [os.path.relpath(abs_path, datadir) for abs_path in
#        glob.glob(os.path.join(datadir, data_model, '*.gpml')) + glob.glob(os.path.join(datadir, data_model, '*.gpmlz'))]
# 2) Explicitly list all the topology files (must be relative to the 'data/' directory).
#topology_filenames = [
#  data_model + '/topology_file1.gpml',
#  data_model + '/topology_file2.gpml',
#]
if (data_model == 'Global_1000-0_Model_2017' or
    data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction')):
    # There are other GPML files that we don't need to include.
    topology_filenames = [
        data_model + '/250-0_plate_boundaries_Merdith_et_al.gpml',
        data_model + '/410-250_plate_boundaries_Merdith_et_al.gpml',
        data_model + '/1000-410-Convergence_Merdith_et_al.gpml',
        data_model + '/1000-410-Divergence_Merdith_et_al.gpml',
        data_model + '/1000-410-Topologies_Merdith_et_al.gpml',
        data_model + '/1000-410-Transforms_Merdith_et_al.gpml',
        data_model + '/TopologyBuildingBlocks_Merdith_et_al.gpml',
    ]
elif '1.8ga' in data_model.lower():
    # There are other GPML files that we don't need to include.
    topology_filenames = [
        data_model + '/250-0_plate_boundaries.gpml',
        data_model + '/410-250_plate_boundaries.gpml',
        data_model + '/1000-410-Convergence.gpml',
        data_model + '/1000-410-Divergence.gpml',
        data_model + '/1000-410-Topologies.gpml',
        data_model + '/1000-410-Transforms.gpml',
        data_model + '/TopologyBuildingBlocks.gpml',
        data_model + '/1800-1000_plate_boundaries.gpml',
    ]
else:
    topology_filenames = [os.path.relpath(abs_path, datadir) for abs_path in
            glob.glob(os.path.join(datadir, data_model, '*.gpml')) + glob.glob(os.path.join(datadir, data_model, '*.gpmlz'))]

# The continental polygons file (relative to the 'data/' directory) used for plate velocity calculations (when plate velocity is enabled).
# NOTE: Set to None to use topologies instead (which includes continental and oceanic crust).
if data_model.startswith('Global_Model_WD_Internal_Release'):
    plate_velocity_continental_polygons_file = data_model + '/StaticGeometries/ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentsOnly.shp'
elif (data_model == 'Global_1000-0_Model_2017' or
      data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction')):
    plate_velocity_continental_polygons_file = data_model + '/shapes_continents_Merdith_et_al.gpml'
elif data_model == 'Zahirovic_etal_2022_GDJ':
    plate_velocity_continental_polygons_file = data_model + '/StaticGeometries/ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentalPolygons.shp'
elif '1.8ga' in data_model.lower():
    plate_velocity_continental_polygons_file = data_model + '/shapes_continents.gpmlz'
else:
    plate_velocity_continental_polygons_file = None

# The grid spacing (in degrees) between points in the grid used for plate velocity calculations (when plate velocity is enabled).
plate_velocity_grid_spacing = 2.0

# The grid spacing (in degrees) between points in the grid used for contouring/aggregrating blocks of continental polygons.
#
# 2.0 degrees seems almost better than 1.0 or 0.5 (which captures too small detail along continent boundary).
#
# Note: It's probably best to make this the same as 'plate_velocity_grid_spacing' so that the same points used for
#       calculating velocities are used for generating continent contours.
#
# NOTE: This only applies if both plate velocity is enabled (see 'get_plate_velocity_params' below) and
#       'plate_velocity_continental_polygons_file' is specified (ie, not None).
plate_velocity_continental_fragmentation_point_spacing_degrees = 2.0

# Contour polygons smaller than this will be excluded when contouring/aggregrating blocks of continental polygons.
# Note: Units here are for normalised sphere (ie, steradians or square radians) so full Earth area is 4*pi.
#       So 0.1 covers an area of approximately 4,000,000 km^2 (ie, 0.1 * 6371^2, where Earth radius is 6371km).
#
# Note: Currently we set this to zero and instead rely on the plate velocity cost function below to perform
#       thresholding based on each contour's perimeter/area ratio.
#
# NOTE: This only applies if both plate velocity is enabled (see 'get_plate_velocity_params' below) and
#       'plate_velocity_continental_polygons_file' is specified (ie, not None).
def plate_velocity_continental_fragmentation_area_threshold_steradians(time):
    return 0.0

# Gaps between continent polygons smaller than this will be excluded when contouring/aggregrating blocks of continental polygons.
# Note: Units here are for normalised sphere (ie, radians).
#       So 1.0 radian is approximately 6371 km (where Earth radius is 6371 km).
#
# NOTE: This only applies if both plate velocity is enabled (see 'get_plate_velocity_params' below) and
#       'plate_velocity_continental_polygons_file' is specified (ie, not None).
def plate_velocity_continental_fragmentation_gap_threshold_radians(time):
    if (data_model == 'Global_1000-0_Model_2017' or
        data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction') or
        data_model.startswith('Global_Model_WD_Internal_Release') or
        data_model == 'Zahirovic_etal_2022_GDJ' or
        '1.8ga' in data_model.lower()):
        if time < 200:
            return math.radians(0.0)  # 1 degree is about 110 km
        elif time < 400:
            return math.radians(1.0)  # 1 degree is about 110 km
        else:
            return math.radians(2.0)  # 1 degree is about 110 km
    else:
        return math.radians(2.0)  # 1 degree is about 110 km


# Temporary: Allow input of GPlates exported net rotation file.
# TODO: Remove when we can calculate net rotation in pygplates for a deforming model (like GPlates can).
#       Currently we ignore deforming networks when calculating net rotation since we also ignore them for plate velocities.
#gplates_net_rotation_filename = data_model + '/optimisation/total-net-rotations.csv'
gplates_net_rotation_filename = None

if data_model.startswith('Global_Model_WD_Internal_Release'):
    ridge_file = data_model + '/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_Ridges.gpml'
    isochron_file = data_model + '/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_Isochrons.gpml'
    isocob_file = data_model + '/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_IsoCOB.gpml'
elif (data_model == 'Global_1000-0_Model_2017' or
      data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction') or
      data_model == 'Zahirovic_etal_2022_GDJ' or
      '1.8ga' in data_model.lower()):
    #
    # For (data_model == 'Global_1000-0_Model_2017') or (data_model == 'Muller++_2015_AREPS_CORRECTED') ...
    #
    ##################################################################################################################################
    #
    # There are no static geometries (besides coastlines) for this data model.
    #
    # NOTE: SO USING SAME FILES AS 'Global_Model_WD_Internal_Release_2019_v2'.
    #       THIS IS OK IF WE'RE NOT INCLUDING FRACTURE ZONES (BECAUSE THEN THESE FILES ARE NOT USED FOR FINAL OPTIMISED ROTATIONS).
    #
    ##################################################################################################################################
    ridge_file = 'Global_Model_WD_Internal_Release_2019_v2/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_Ridges_2019_v2.gpml'
    isochron_file = 'Global_Model_WD_Internal_Release_2019_v2/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_Isochrons_2019_v2.gpml'
    isocob_file = 'Global_Model_WD_Internal_Release_2019_v2/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_IsoCOB_2019_v2.gpml'
else:
    #
    # Original files used in original optimisation script...
    #
    ridge_file = 'Global_EarthByte_230-0Ma_GK07_AREPS_Ridges.gpml'
    isochron_file = 'Global_EarthByte_230-0Ma_GK07_AREPS_Isochrons.gpmlz'
    isocob_file = 'Global_EarthByte_230-0Ma_GK07_AREPS_IsoCOB.gpml'


#
# Which components are enabled and their weightings and cost function and optional restricted bounds.
#
# Each return value is a 4-tuple:
#  1. Enable boolean (True or False),
#  2. Weight value (float),
#  3. Cost function (function accepting parameters that are specific to the component being optimised),
#  4. Optional restricted bounds (2-tuple of min/max cost, or None).
#
# For restricted bounds, use None if you are not restricting.
# Otherwise use a (min, max) tuple.
#
# NOTE: The weights are inverse weights (ie, the constraint costs are *multiplied* by "1.0 / weight").
#
def get_fracture_zone_params(age):
    # Cost function - see "objective_function.py" for definition of function arguments...
    def cost_function(fz):
        # NOTE: Import any modules used in this function here
        #       (since this function's code might be serialised over the network to remote nodes).
        return fz[0] + fz[1]

    # Disable fracture zones.
    return False, 1.0, cost_function, None

def get_net_rotation_params(age):
    # Cost function - see "objective_function.py" for definition of function arguments...
    def cost_function(PTLong1, PTLat1, PTangle1, SPLong, SPLat, SPangle, SPLong_NNR, SPLat_NNR, SPangle_NNR, nr_over_interval):
        # NOTE: Import any modules used in this function here
        #       (since this function's code might be serialised over the network to remote nodes).
        import numpy as np
        return nr_over_interval + np.mean(np.abs(PTangle1))

    # Note: Use units of degrees/Myr...
    #nr_bounds = (0.08, 0.20)
    
    if (data_model == 'Global_1000-0_Model_2017' or
        data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction') or
        data_model.startswith('Global_Model_WD_Internal_Release') or
        data_model == 'Zahirovic_etal_2022_GDJ' or
        '1.8ga' in data_model.lower()):
        nr_bounds = (0.08, 0.20)
        if age <= 80:
            return  True, 1.0, cost_function, nr_bounds  # Weight is always 1.0 for 0-80Ma
        else:
            return  True, 1.0, cost_function, nr_bounds  # 1.0 gives a *multiplicative* weight of 1.0
    else:
        return True, 1.0, cost_function, None

def get_trench_migration_params(age):
    # Cost function - see "objective_function.py" for definition of function arguments...
    def cost_function(trench_vel, trench_obl, tm_vel_orth, tm_mean_vel_orth, tm_mean_abs_vel_orth):
        # NOTE: Import any modules used in this function here
        #       (since this function's code might be serialised over the network to remote nodes).
        import numpy as np

        # trench_numTotal = len(tm_vel_orth)
        # trench_numRetreating = len(np.where(tm_vel_orth > 0)[0])
        # trench_numAdvancing = len(tm_vel_orth) - trench_numRetreating
        # trench_percent_retreat = round((np.float(trench_numRetreating) / np.float(trench_numTotal)) * 100, 2)
        # trench_percent_advance = 100. - trench_percent_retreat
        # trench_sumAbsVel_n = np.sum(np.abs(tm_vel_orth)) / len(tm_vel_orth)
        # trench_numOver30 = len(np.where(tm_vel_orth > 30)[0])
        # trench_numLessNeg30 = len(np.where(tm_vel_orth < -30)[0])

        # Calculate cost
        #tm_eval_1 = trench_percent_advance * 10
        #tm_eval_2 = trench_sumAbsVel_n * 15

        # 1. trench percent advance + trench abs vel mean
        #tm_eval = (tm_eval_1 + tm_eval_2) / 2

        # 2. trench_abs_vel_mean orthogonal
        tm_eval_2 = tm_mean_abs_vel_orth

        # 3. number of trenches in advance
        #tm_eval_3 = trench_numAdvancing * 2

        # 4. abs median
        #tm_eval = np.median(abs(np.array(tm_vel_orth)))

        # 5. standard deviation
        tm_eval_5 = np.std(tm_vel_orth)

        # 6. variance
        #tm_stats = stats.describe(tm_vel_orth)
        #tm_eval_6 = tm_stats.variance

        # 7. trench absolute motion abs vel mean
        #tm_eval_7 = (np.sum(np.abs(trench_vel)) / len(trench_vel)) * 15

        tm_eval = (tm_eval_2 + tm_eval_5) * 3
        
        # Original equation
        #tm_eval = ((tm_eval_5 * (trench_numRetreating * trench_sumAbsVel_n)) / \
        #           (trench_numTotal - (trench_numOver30 + trench_numLessNeg30)))
        
        #tm_eval = (tm_eval_2 + tm_eval_5 + trench_numAdvancing) / trench_numRetreating

        return tm_eval
        
    # Note: Use units of mm/yr (same as km/Myr)...
    #tm_bounds = [0, 30]
    
    if (data_model == 'Global_1000-0_Model_2017' or
        data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction') or
        data_model.startswith('Global_Model_WD_Internal_Release') or
        data_model == 'Zahirovic_etal_2022_GDJ' or
        '1.8ga' in data_model.lower()):
        # # Override default cost function for 1Ga model - see "objective_function.py" for definition of function arguments...
        # def cost_function(trench_vel, trench_obl, tm_vel_orth, tm_mean_vel_orth, tm_mean_abs_vel_orth):
        #     # NOTE: Import any modules used in this function here
        #     #       (since this function's code might be serialised over the network to remote nodes).
        #     import numpy as np
        #     return 16.0 * np.mean(np.abs(np.where(tm_vel_orth > 0, 0.1 * tm_vel_orth, tm_vel_orth)))
        
        tm_bounds = [-30, 30]
        if age <= 80:
            return True, 1.0, cost_function, tm_bounds  # Weight is always 1.0 for 0-80Ma
        else:
            # NOTE: These are inverse weights (ie, the constraint costs are *multiplied* by "1.0 / weight").
            return True, 2.0, cost_function, tm_bounds  # 2.0 gives a *multiplicative* weight of 0.5
    else:
        return True, 1.0, cost_function, None

def get_hotspot_trail_params(age):
    # Cost function - see "objective_function.py" for definition of function arguments...
    def cost_function(hs, distance_median, distance_sd):
        # NOTE: Import any modules used in this function here
        #       (since this function's code might be serialised over the network to remote nodes).
        return distance_median + distance_sd

    # Only use hotspot trails for 0-80Ma.
    if age <= 80:
        return True, 1.0, cost_function, None  # Weight is always 1.0 for 0-80Ma
    else:
        return False, 1.0, cost_function, None

def get_plate_velocity_params(age):
    # Cost function - see "objective_function.py" for definition of function arguments...
    def cost_function(plate_features_are_topologies, velocity_vectors_in_contours, ref_rotation_start_age):
        # NOTE: Import any modules used in this function here
        #       (since this function's code might be serialised over the network to remote nodes).
        import numpy as np

        if plate_features_are_topologies:
            #total_plate_perimeter = 0.0
            #total_plate_area = 0.0

            # Calculate median of all velocities (in all contours/plates).
            velocity_magnitudes = []
            for plate_perimeter, plate_area, velocity_vectors_in_contour in velocity_vectors_in_contours:
                velocity_magnitudes.extend(velocity_vector.get_magnitude() for velocity_vector in velocity_vectors_in_contour)

                #total_plate_perimeter += plate_perimeter
                #total_plate_area += plate_area
            
            # If there were no contours then no need to penalize, so return zero cost.
            if not velocity_magnitudes:
                return 0.0
            
            return np.median(velocity_magnitudes)

        else:  # continent contours...

            # If there were no contours at all then just return zero cost.
            #
            # This shouldn't happen because we should be getting all contours (except below min area threshold),
            # because none should be excluded based on perimeter/area ratio.
            #
            # And note that returning zero cost is generally not a good idea because if the plate velocity cost happens
            # to be the only cost metric (eg, net rotation and trench migration weights are zero) then the optimizer cannot
            # find a minimum (since it's likely all model seeds could return zero). In this case the optimizer could just return
            # any solution. This was observed (when contours below a threshold were given zero weight, which we no longer do)
            # as very high speed absolute plate motions that zipped back and forth across the globe.
            if not velocity_vectors_in_contours:
                return 0.0
            
            # Calculate median of all velocities (in all contours).
            velocity_magnitudes = []
            for _, _, velocity_vectors_in_contour in velocity_vectors_in_contours:
                velocity_magnitudes.extend(velocity_vector.get_magnitude() for velocity_vector in velocity_vectors_in_contour)
            median_velocity = np.median(velocity_magnitudes)

            return median_velocity

    # Note: Use units of mm/yr (same as km/Myr)...
    #pv_bounds = [0, 60]
    
    if (data_model == 'Global_1000-0_Model_2017' or
        data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction') or
        data_model.startswith('Global_Model_WD_Internal_Release') or
        data_model == 'Zahirovic_etal_2022_GDJ' or
        '1.8ga' in data_model.lower()):
        pv_bounds = [0, 60]
        if age <= 80:
            return True, 1.0, cost_function, pv_bounds  # Weight is always 1.0 for 0-80Ma
        else:
            return True, 2.0, cost_function, pv_bounds  # 2.0 gives a *multiplicative* weight of 0.5
    else:
        return True, 1.0, cost_function, None


#
# Which reference plate ID and rotation file to use at a specific age.
#
USE_NNR_REFERENCE_FRAME = 0
USE_OPTIMISED_REFERENCE_FRAME = 1
def get_reference_params(age):
    """
    Returns a 2-tuple containg reference plate ID and reference rotation filename (or None).
    
    If reference rotation filename is None then it means the no-net-rotation model should be used.
    """
    if (data_model == 'Global_1000-0_Model_2017' or
        data_model.startswith('SM2-Merdith_et_al_1_Ga_reconstruction') or
        data_model.startswith('Global_Model_WD_Internal_Release') or
        data_model == 'Zahirovic_etal_2022_GDJ' or
        '1.8ga' in data_model.lower()):
        # Choose NNR, Optimsed or Africa reference frame.
        reference_frame = USE_OPTIMISED_REFERENCE_FRAME

        if reference_frame == USE_NNR_REFERENCE_FRAME:
            ref_rotation_file = USE_NNR_REFERENCE_FRAME
            ref_rotation_plate_id = 5  # Use optimised absolute reference frame
        elif reference_frame == USE_OPTIMISED_REFERENCE_FRAME:
            ref_rotation_file = USE_OPTIMISED_REFERENCE_FRAME
            if age <= 550:
                ref_rotation_plate_id = 701
            else:
                ref_rotation_plate_id = 101
        else:
            if age <= 550:
                ref_rotation_plate_id = 701
                ref_rotation_file = 'Global_1000-0_Model_2017/pmag/550_0_Palaeomagnetic_Africa_S.rot'
            else:
                ref_rotation_plate_id = 101
                ref_rotation_file = 'Global_1000-0_Model_2017/pmag/1000_550_Laurentia_pmag_reference.rot'
    else:
        ref_rotation_plate_id = 701
        ref_rotation_file = 'Palaeomagnetic_Africa_S.rot'
    
    return ref_rotation_plate_id, ref_rotation_file


search = "Initial"
# If True then temporarily expand search radius to 180 whenever the reference plate changes.
# Normally the reference plate stays constant at Africa (701), but does switch to 101 for the 1Ga model.
# It's off by default since it doesn't appear to change the results, and may sometimes cause job to fail
# on Artemis (presumably since 'models' is increased by a factor of 2.5) - although problem manifested
# as failure to read the rotation file being optimised, so it was probably something else.
expand_search_radius_on_ref_plate_switches = False
rotation_uncertainty = 180
auto_calc_ref_pole = True

model_stop_condition = 'threshold'
max_iter = 5  # Only applies if model_stop_condition != 'threshold'


# Trench migration parameters
tm_method = 'pygplates' # 'pygplates' for new method OR 'convergence' for old method
tm_data_type = data_model


# Hotspot parameters:
interpolated_hotspot_trails = True
use_trail_age_uncertainty = True

# Millions of years - e.g. 2 million years @ 50mm per year = 100km radius uncertainty ellipse
trail_age_uncertainty_ellipse = 1

include_chains = ['Louisville', 'Tristan', 'Reunion', 'St_Helena', 'Foundation', 'Cobb', 'Samoa', 'Tasmantid', 
                  'Hawaii']
#include_chains = ['Louisville', 'Tristan', 'Reunion', 'Hawaii', 'St_Helena', 'Tasmantid']


# Large area grid search to find minima
if search == 'Initial':

    #search_type = 'Random'
    search_type = 'Uniform'

# Uses grid search minima as seed for targeted secondary search (optional)
elif search == 'Secondary':

    search_type = 'Uniform'
    search_radius = 15
    rotation_uncertainty = 30

    models = 60
    auto_calc_ref_pole = False

# Used when auto_calc_ref_pole is False.
no_auto_ref_rot_longitude = -53.5
no_auto_ref_rot_latitude = 56.6
no_auto_ref_rot_angle = -2.28

interpolation_resolution = 5
rotation_age_of_interest = True

hst_file = 'HotspotTrails.geojson'
hs_file = 'HotspotCatalogue2.geojson'
interpolated_hotspots = 'interpolated_hotspot_chains_5Myr.xlsx'


# Don't plot in this workflow.
# This is so it can be run on an HPC cluster with no visualisation node.
plot = False


#
# How to handle warnings.
#
def warning_format(message, category, filename, lineno, file=None, line=None):
    # return '{0}:{1}: {1}:{1}\n'.format(filename, lineno, category.__name__, message)
    return '{0}: {1}\n'.format(category.__name__, message)
# Print the warnings without the filename and line number.
# Users are not going to want to see that.
warnings.formatwarning = warning_format

# Always print warnings (not just the first time a particular message is encountered at a particular location).
#warnings.simplefilter("always")
