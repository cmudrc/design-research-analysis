"""Internal backend adapters for sequence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SEQ_IMPORT_ERROR = (
    "Sequence modeling requires optional sequence dependencies. "
    "Install with `pip install design-research-analysis[seq]`."
)


class _BaseHMMBackend:
    """Interface for HMM backend adapters."""

    name: str

    def create_gaussian_hmm(
        self,
        *,
        n_states: int,
        covariance_type: str,
        n_iter: int,
        seed: int,
    ) -> Any:
        """Create a Gaussian HMM model instance."""
        raise NotImplementedError

    def create_discrete_hmm(
        self,
        *,
        n_states: int,
        n_iter: int,
        seed: int,
        n_symbols: int,
    ) -> Any:
        """Create a discrete-emission HMM model instance."""
        raise NotImplementedError


@dataclass(slots=True)
class HmmlearnBackend(_BaseHMMBackend):
    """hmmlearn-backed implementation of the backend adapter interface."""

    name: str = "hmmlearn"

    def _load_hmm_module(self) -> Any:
        try:
            from hmmlearn import hmm
        except ImportError as exc:
            raise ImportError(_SEQ_IMPORT_ERROR) from exc
        return hmm

    def create_gaussian_hmm(
        self,
        *,
        n_states: int,
        covariance_type: str,
        n_iter: int,
        seed: int,
    ) -> Any:
        """Create a configured ``hmmlearn.GaussianHMM`` instance."""
        hmm = self._load_hmm_module()
        return hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
        )

    def create_discrete_hmm(
        self,
        *,
        n_states: int,
        n_iter: int,
        seed: int,
        n_symbols: int,
    ) -> Any:
        """Create a configured ``hmmlearn.CategoricalHMM`` instance."""
        hmm = self._load_hmm_module()
        if not hasattr(hmm, "CategoricalHMM"):
            raise ImportError(
                "Installed hmmlearn does not provide CategoricalHMM. "
                'Install a newer version with `pip install "design-research-analysis[seq]"`.'
            )

        model = hmm.CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            random_state=seed,
            init_params="ste",
            params="ste",
        )
        model.n_features = n_symbols
        return model


def get_hmm_backend(name: str = "hmmlearn") -> _BaseHMMBackend:
    """Return an initialized HMM backend adapter.

    Args:
        name: Backend name. ``hmmlearn`` is currently supported.

    Returns:
        Backend adapter instance.

    Raises:
        ValueError: If ``name`` is not a supported backend.
    """
    if name == "hmmlearn":
        return HmmlearnBackend()
    raise ValueError("Unsupported HMM backend. Valid options: hmmlearn")
