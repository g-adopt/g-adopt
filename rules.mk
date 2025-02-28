# if we're running from the CI, we want to run everything
# with tsp, echo the job on start and finish, and
# set a timeout

RUN_CMD = $(if $(BATCH_MODE),tsp $(if $(category),-L $(category),))

run-%: %.py
	$(RUN_CMD) $(if $(filter-out 1,$(ncpus)),mpiexec -np $(ncpus)) python3 $< $(PETSC_FLAGS)


# define these as expansion rules so they can be used to refer to the calling case
current_dir = $(notdir $(patsubst %/,%,$(CURDIR)))
test_class = $(notdir $(patsubst %/,%,$(dir $(CURDIR))))
