.PHONY: pysyrenn_coverage start_server acas_lines_experiment \
        integrated_gradients_experiment linearity_hypothesis_experiment \
        exactline_experiments toy_examples_experiment acas_planes_experiment \
	model_checking_experiment plane_experiments \
	all_experiments

pysyrenn_coverage:
	bazel test //pysyrenn/...
	bazel run coverage_report

start_server:
	bazel run syrenn_server:server --define BAZEL_PYTHON_ONLY_WARN=true

# Experiments from [1]
acas_lines_experiment:
	@echo "Running the ACAS Xu Lines experiment"
	bazel run experiments:acas_lines

integrated_gradients_experiment:
	@echo "Running the Integrated Gradinets experiment"
	bazel run experiments:integrated_gradients

linearity_hypothesis_experiment:
	@echo "Running the Linearity Hypothesis experiment"
	bazel run experiments:linearity_hypothesis

exactline_experiments: acas_lines_experiment integrated_gradients_experiment \
		       linearity_hypothesis_experiment

# Experiments from [2]
toy_examples_experiment:
	@echo "Running the Toy Examples experiment"
	bazel run experiments:toy_examples

acas_planes_experiment:
	@echo "Running the ACAS Xu Planes experiment"
	bazel run experiments:acas_planes

model_checking_experiment:
	@echo "Running the Bounded Model Checking experiment"
	bazel run experiments:model_checking

plane_experiments: toy_examples_experiment acas_planes_experiment \
		   model_checking_experiment

# Run experiments from [1] and [2]
all_experiments: exactline_experiments plane_experiments
