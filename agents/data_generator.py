import pandas as pd
import numpy as np

class SyntheticDataGenerator:
    def generate(self, drift_report: dict, n_samples: int = 500):
        """
        Generate realistic synthetic training samples based on drifted feature distributions.
        Ensures domain validity (no negative ages, no negative balances, realistic incomes).
        """
        psi = drift_report.get("psi_by_feature", {})

        df = pd.DataFrame()

        # === AGE ===
        if psi.get("age", 0) > 0.1:
            # Drifted age → shift mean upward
            df["age"] = np.random.normal(loc=40 + 10 * psi["age"], scale=8, size=n_samples)
        else:
            df["age"] = np.random.normal(loc=40, scale=8, size=n_samples)

        df["age"] = df["age"].clip(lower=18, upper=90)

        # === INCOME ===
        if psi.get("income", 0) > 0.1:
            # Drifted income → shift log-normal mean upward
            df["income"] = np.random.lognormal(mean=10 + psi["income"], sigma=0.3, size=n_samples)
        else:
            df["income"] = np.random.lognormal(mean=10, sigma=0.3, size=n_samples)

        # === BALANCE ===
        if psi.get("balance", 0) > 0.1:
            df["balance"] = np.random.normal(loc=2000 + 500 * psi["balance"], scale=300, size=n_samples)
        else:
            df["balance"] = np.random.normal(loc=2000, scale=300, size=n_samples)

        df["balance"] = df["balance"].clip(lower=0)

        # === DEFAULT FLAG (binary) ===
        # Higher income → lower default probability
        prob_default = 1 / (1 + np.exp((df["income"] - df["income"].median()) / 5000))
        df["default"] = (np.random.rand(n_samples) < prob_default).astype(int)

        # === LABEL (binary) ===
        # Simple rule: high income + high balance → label 1
        score = (
            0.4 * (df["income"] / df["income"].max()) +
            0.3 * (df["balance"] / df["balance"].max()) +
            0.3 * (df["age"] / df["age"].max())
        )
        df["label"] = (score > score.median()).astype(int)

        return df