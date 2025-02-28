# running in batch mode, we defer scheduling to tsp
# otherwise, we just run directly
# in both cases, we print some timing information
run_regular = d=$$(date +%s); $(1) && echo "$(2) took $$(($$(date +%s)-d)) seconds"
run_batch = id=$$(tsp -f $(if $(category),-L $(category)) $(1)) && echo "[$(category)]$(2) took $$(tsp -i $$id | sed -n 's/Time run: //p')"
run_cmd = $(if $(BATCH_MODE),$(run_batch),$(run_regular))

exec_cmd = $(if $(filter-out 1,$(ncpus)),mpiexec -np $(ncpus)) python3 $< $(PETSC_FLAGS)

# running in batch mode (i.e. from the CI) we want to set a timeout
# so that we can get useful output without hitting the GitHub Actions
# hard limit of 6 hours
exec_wrapper = $(if $(BATCH_MODE),timeout 2h $(exec_cmd),$(exec_cmd))

run-%: %.py
	@$(call run_cmd,$(exec_wrapper),$(or $(desc),$<))


# define these as expansion rules so they can be used to refer to the calling case
current_dir = $(notdir $(patsubst %/,%,$(CURDIR)))
test_class = $(notdir $(patsubst %/,%,$(dir $(CURDIR))))
