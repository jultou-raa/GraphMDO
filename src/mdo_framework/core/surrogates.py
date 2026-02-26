import numpy as np
from smt.surrogate_models import KRG, KPLS
from smt.applications import MFK


class SMTSurrogate:
    """
    Wrapper for SMT surrogate models, supporting single and multi-fidelity.
    """

    def __init__(self, model_type="KRG", multi_fidelity=False):
        self.model_type = model_type
        self.multi_fidelity = multi_fidelity
        self.model = None

    def train(
        self,
        xt: np.ndarray,
        yt: np.ndarray,
        x_lf: np.ndarray = None,
        y_lf: np.ndarray = None,
    ):
        """
        Trains the surrogate model.

        Args:
            xt: High-fidelity training inputs.
            yt: High-fidelity training outputs.
            x_lf: Low-fidelity training inputs (for multi-fidelity).
            y_lf: Low-fidelity training outputs (for multi-fidelity).
        """
        n_dims = xt.shape[1]

        if self.multi_fidelity:
            self.model = MFK(theta0=[1e-2] * n_dims, print_global=False)

            if x_lf is not None and y_lf is not None:
                self.model.set_training_values(x_lf, y_lf, name=0)

            self.model.set_training_values(xt, yt)

        else:
            if self.model_type == "KRG":
                self.model = KRG(theta0=[1e-2] * n_dims, print_global=False)
            elif self.model_type == "KPLS":
                # KPLS reduces dimensions. n_comp defaults to 1.
                # theta0 must match n_comp, NOT n_dims.
                # Default n_comp=1 in SMT KPLS.
                n_comp = 1
                self.model = KPLS(
                    theta0=[1e-2] * n_comp, n_comp=n_comp, print_global=False
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model.set_training_values(xt, yt)

        self.model.train()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts values at x."""
        return self.model.predict_values(x)

    def predict_variances(self, x: np.ndarray) -> np.ndarray:
        """Predicts variances at x."""
        return self.model.predict_variances(x)
