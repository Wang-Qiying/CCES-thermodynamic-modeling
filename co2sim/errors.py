class ModelError(RuntimeError):
    kind = "model_error"


class PropertyFailure(ModelError):
    kind = "property_failure"


class NoPhysicalSolution(ModelError):
    kind = "no_physical_solution"


class NumericalFailure(ModelError):
    kind = "numerical_failure"


class ModeInapplicable(ModelError):
    kind = "mode_inapplicable"
