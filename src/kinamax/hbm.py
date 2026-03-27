"""Helpers for Harmonic Balance Method (HBM) coefficient manipulations.

The real Fourier coefficient layout used across this module is:

    X = [a0, a1, ..., aN, b1, ..., bN]

which represents the real periodic signal

    x(t) = a0 + sum_n (a_n cos(n * wd * t) + b_n sin(n * wd * t))
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .core import namedtuple_repr

__all__ = [
    "FourierCoeffs",
    "SampledSignal",
    "add_fourier_coeffs",
    "sub_fourier_coeffs",
    "scale_fourier_coeffs",
    "sum_fourier_coeffs",
    "add_sampled_signals",
    "sub_sampled_signals",
    "scale_sampled_signal",
    "sum_sampled_signals",
    "coeffs_to_complex",
    "complex_to_coeffs",
    "coeffs_derivative",
    "coeffs_to_table",
    "time_grid",
    "coeffs_to_time_signal",
    "time_to_coeffs",
    "coeffs_pow",
]

class FourierCoeffs(NamedTuple):
    """Real Fourier coefficients paired with the fundamental frequency in Hz.

    The arithmetic implemented on this container is intentionally limited to the
    linear operations that preserve the Fourier basis directly:

    - addition and subtraction between coefficient vectors
    - unary plus and minus
    - multiplication and division by scalars

    These operations preserve `frequency` without checking it at runtime so they
    remain compatible with `jax.jit`. In practice, matching frequencies are a
    contract of use.

    Static constructors are also provided for the common forcing patterns used
    in HBM models:

    - `FourierCoeffs.zeros(...)`
    - `FourierCoeffs.cosine(...)`
    - `FourierCoeffs.sine(...)`
    - `FourierCoeffs.phased(...)`

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = FourierCoeffs(
    ...     values=jnp.array([0.0, 1.0, 0.5]),
    ...     frequency=jnp.float32(5.0),
    ... )
    >>> print(coeffs)
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (3,)   ┆ float32 ┆ [0. , 1. , 0.5]...
    ...
    >>> print(coeffs + FourierCoeffs(values=jnp.array([1.0, 0.0, 0.0]), frequency=5.0))
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (3,)   ┆ float32 ┆ [1. , 1. , 0.5]...
    ...
    """

    values: jax.Array
    frequency: float | jax.Array

    def __repr__(self) -> str:
        return namedtuple_repr(
            "FourierCoeffs",
            {
                "frequency": self.frequency,
                "values": self.values,
            },
        )

    @staticmethod
    def zeros(order: int, frequency: ArrayLike) -> "FourierCoeffs":
        """Build a zero coefficient vector up to harmonic ``order``."""
        if order < 0:
            raise ValueError("order must be >= 0.")
        dtype = jnp.result_type(jnp.asarray(frequency), 0.0)
        return FourierCoeffs(
            values=jnp.zeros(2 * order + 1, dtype=dtype),
            frequency=jnp.asarray(frequency),
        )

    @staticmethod
    def cosine(
        amplitude: ArrayLike,
        harmonic: int,
        order: int,
        frequency: ArrayLike,
    ) -> "FourierCoeffs":
        """Build ``amplitude * cos(harmonic * 2*pi*frequency*t)``."""
        if order < 0:
            raise ValueError("order must be >= 0.")
        if harmonic < 1 or harmonic > order:
            raise ValueError("harmonic must satisfy 1 <= harmonic <= order.")
        coeffs = FourierCoeffs.zeros(order=order, frequency=frequency)
        return FourierCoeffs(
            values=coeffs.values.at[harmonic].set(jnp.asarray(amplitude)),
            frequency=coeffs.frequency,
        )

    @staticmethod
    def sine(
        amplitude: ArrayLike,
        harmonic: int,
        order: int,
        frequency: ArrayLike,
    ) -> "FourierCoeffs":
        """Build ``amplitude * sin(harmonic * 2*pi*frequency*t)``."""
        if order < 0:
            raise ValueError("order must be >= 0.")
        if harmonic < 1 or harmonic > order:
            raise ValueError("harmonic must satisfy 1 <= harmonic <= order.")
        coeffs = FourierCoeffs.zeros(order=order, frequency=frequency)
        return FourierCoeffs(
            values=coeffs.values.at[order + harmonic].set(jnp.asarray(amplitude)),
            frequency=coeffs.frequency,
        )

    @staticmethod
    def phased(
        amplitude: ArrayLike,
        phase: ArrayLike,
        harmonic: int,
        order: int,
        frequency: ArrayLike,
    ) -> "FourierCoeffs":
        """Build ``amplitude * cos(harmonic * 2*pi*frequency*t + phase)``."""
        amplitude = jnp.asarray(amplitude)
        phase = jnp.asarray(phase)
        return (
            FourierCoeffs.cosine(
                amplitude=amplitude * jnp.cos(phase),
                harmonic=harmonic,
                order=order,
                frequency=frequency,
            )
            - FourierCoeffs.sine(
                amplitude=amplitude * jnp.sin(phase),
                harmonic=harmonic,
                order=order,
                frequency=frequency,
            )
        )

    def __pos__(self) -> "FourierCoeffs":
        return self

    def __neg__(self) -> "FourierCoeffs":
        return FourierCoeffs(values=-jnp.asarray(self.values), frequency=self.frequency)

    def __add__(self, other: object) -> "FourierCoeffs":
        if not isinstance(other, FourierCoeffs):
            return NotImplemented
        return FourierCoeffs(
            values=jnp.asarray(self.values) + jnp.asarray(other.values),
            frequency=self.frequency,
        )

    def __radd__(self, other: object) -> "FourierCoeffs":
        if other == 0:
            return self
        if not isinstance(other, FourierCoeffs):
            return NotImplemented
        return other + self

    def __sub__(self, other: object) -> "FourierCoeffs":
        if not isinstance(other, FourierCoeffs):
            return NotImplemented
        return FourierCoeffs(
            values=jnp.asarray(self.values) - jnp.asarray(other.values),
            frequency=self.frequency,
        )

    def __rsub__(self, other: object) -> "FourierCoeffs":
        if not isinstance(other, FourierCoeffs):
            return NotImplemented
        return other - self

    def __mul__(self, scalar: ArrayLike) -> "FourierCoeffs":
        return FourierCoeffs(
            values=jnp.asarray(self.values) * jnp.asarray(scalar),
            frequency=self.frequency,
        )

    def __rmul__(self, scalar: ArrayLike) -> "FourierCoeffs":
        return self * scalar

    def __truediv__(self, scalar: ArrayLike) -> "FourierCoeffs":
        return FourierCoeffs(
            values=jnp.asarray(self.values) / jnp.asarray(scalar),
            frequency=self.frequency,
        )


class SampledSignal(NamedTuple):
    """Uniformly sampled signal over one period, tagged by its frequency in Hz.

    The arithmetic implemented on this container is limited to the linear
    operations that preserve the sampling layout directly:

    - addition and subtraction between sampled signals
    - unary plus and minus
    - multiplication and division by scalars

    As for `FourierCoeffs`, `frequency` is propagated without runtime checks so
    the operations remain compatible with `jax.jit`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> signal = SampledSignal(
    ...     values=jnp.array([0.0, 1.0, 0.0, -1.0]),
    ...     frequency=jnp.float32(5.0),
    ... )
    >>> print(signal)
    SampledSignal
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (4,)   ┆ float32 ┆ [ 0.,  1.,  0., -1.]...
    ...
    >>> print(signal.time_step)
    0.05
    >>> print(signal.time_grid)
    [0.   0.05 0.1  0.15]
    """

    values: jax.Array
    frequency: float | jax.Array

    @property
    def time_step(self) -> jax.Array:
        """Return the uniform time step over one period."""
        values = jnp.asarray(self.values)
        sample_count = values.shape[-1] if values.ndim else 1
        return 1.0 / (jnp.asarray(self.frequency) * sample_count)

    @property
    def time_grid(self) -> jax.Array:
        """Return the evenly spaced time grid over one period."""
        values = jnp.asarray(self.values)
        sample_count = values.shape[-1] if values.ndim else 1
        return jnp.arange(sample_count) * self.time_step[..., None]

    def __repr__(self) -> str:
        return namedtuple_repr(
            "SampledSignal",
            {
                "frequency": self.frequency,
                "values": self.values,
            },
        )

    def __pos__(self) -> "SampledSignal":
        return self

    def __neg__(self) -> "SampledSignal":
        return SampledSignal(values=-jnp.asarray(self.values), frequency=self.frequency)

    def __add__(self, other: object) -> "SampledSignal":
        if not isinstance(other, SampledSignal):
            return NotImplemented
        return SampledSignal(
            values=jnp.asarray(self.values) + jnp.asarray(other.values),
            frequency=self.frequency,
        )

    def __radd__(self, other: object) -> "SampledSignal":
        if other == 0:
            return self
        if not isinstance(other, SampledSignal):
            return NotImplemented
        return other + self

    def __sub__(self, other: object) -> "SampledSignal":
        if not isinstance(other, SampledSignal):
            return NotImplemented
        return SampledSignal(
            values=jnp.asarray(self.values) - jnp.asarray(other.values),
            frequency=self.frequency,
        )

    def __rsub__(self, other: object) -> "SampledSignal":
        if not isinstance(other, SampledSignal):
            return NotImplemented
        return other - self

    def __mul__(self, scalar: ArrayLike) -> "SampledSignal":
        return SampledSignal(
            values=jnp.asarray(self.values) * jnp.asarray(scalar),
            frequency=self.frequency,
        )

    def __rmul__(self, scalar: ArrayLike) -> "SampledSignal":
        return self * scalar

    def __truediv__(self, scalar: ArrayLike) -> "SampledSignal":
        return SampledSignal(
            values=jnp.asarray(self.values) / jnp.asarray(scalar),
            frequency=self.frequency,
        )


def add_fourier_coeffs(a: FourierCoeffs, b: FourierCoeffs) -> FourierCoeffs:
    """Add two Fourier coefficient containers.

    This helper is equivalent to `a + b` and is compatible with `jax.jit`.
    Frequency consistency is assumed by contract and is not checked at runtime.
    """
    return a + b


def sub_fourier_coeffs(a: FourierCoeffs, b: FourierCoeffs) -> FourierCoeffs:
    """Subtract two Fourier coefficient containers."""
    return a - b


def scale_fourier_coeffs(X: FourierCoeffs, scalar: ArrayLike) -> FourierCoeffs:
    """Multiply a Fourier coefficient container by a scalar."""
    return X * scalar


def sum_fourier_coeffs(*terms: FourierCoeffs) -> FourierCoeffs:
    """Sum multiple Fourier coefficient containers.

    This helper is meant for building HBM residuals with repeated linear
    combinations inside JAX-compiled code paths.
    """
    if len(terms) == 0:
        raise ValueError("At least one FourierCoeffs term is required.")
    return sum(terms[1:], start=terms[0])


def add_sampled_signals(a: SampledSignal, b: SampledSignal) -> SampledSignal:
    """Add two sampled-signal containers."""
    return a + b


def sub_sampled_signals(a: SampledSignal, b: SampledSignal) -> SampledSignal:
    """Subtract two sampled-signal containers."""
    return a - b


def scale_sampled_signal(x: SampledSignal, scalar: ArrayLike) -> SampledSignal:
    """Multiply a sampled-signal container by a scalar."""
    return x * scalar


def sum_sampled_signals(*terms: SampledSignal) -> SampledSignal:
    """Sum multiple sampled-signal containers."""
    if len(terms) == 0:
        raise ValueError("At least one SampledSignal term is required.")
    return sum(terms[1:], start=terms[0])


def _harmonic_order_count(coeffs: jax.Array) -> int:
    """Return the highest retained harmonic order for a stacked real vector."""
    size = coeffs.size
    if size < 1 or size % 2 == 0:
        raise ValueError("HBM coefficient vectors must have odd length 2*N + 1.")
    return (size - 1) // 2


def _require_positive_factor(name: str, value: int) -> None:
    """Validate integer sampling factors used for FFT domain conversions."""
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")


def _coeff_values(X: ArrayLike | FourierCoeffs) -> jax.Array:
    """Return the raw coefficient vector from an array or a FourierCoeffs object."""
    if isinstance(X, FourierCoeffs):
        return jnp.asarray(X.values)
    return jnp.asarray(X)


def _sample_values(x: ArrayLike | SampledSignal) -> jax.Array:
    """Return the raw sampled signal from an array or a SampledSignal object."""
    if isinstance(x, SampledSignal):
        return jnp.asarray(x.values)
    return jnp.asarray(x)


def _frequency_to_wd(frequency: ArrayLike) -> jax.Array:
    """Convert a frequency in Hz to an angular frequency in rad/s."""
    return 2.0 * jnp.pi * jnp.asarray(frequency)


def coeffs_to_complex(X: ArrayLike | FourierCoeffs) -> jax.Array:
    """Convert stacked real Fourier coefficients to positive-frequency phasors."""
    coeffs = _coeff_values(X)
    N = _harmonic_order_count(coeffs)
    a0 = coeffs[0]
    a = coeffs[1 : N + 1]
    b = coeffs[N + 1 :]
    return jnp.concatenate([a0[None], a - 1j * b])


def complex_to_coeffs(
    C: ArrayLike, frequency: ArrayLike | None = None
) -> jax.Array | FourierCoeffs:
    """Convert positive-frequency phasors back to stacked real coefficients."""
    complex_coeffs = jnp.asarray(C)
    if complex_coeffs.size < 1:
        raise ValueError("At least one complex coefficient is required.")
    a0 = jnp.real(complex_coeffs[0])
    a = jnp.real(complex_coeffs[1:])
    b = -jnp.imag(complex_coeffs[1:])
    real_dtype = jnp.real(complex_coeffs).dtype
    values = jnp.concatenate([a0[None], a, b]).astype(real_dtype)
    if frequency is None:
        return values
    return FourierCoeffs(values=values, frequency=jnp.asarray(frequency))


def coeffs_derivative(
    X: ArrayLike | FourierCoeffs, wd: ArrayLike | None = None, order: int = 1
) -> jax.Array | FourierCoeffs:
    """Differentiate a coefficient vector with respect to time.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = FourierCoeffs(values=jnp.array([0.0, 1.0, 0.0]), frequency=5.0)
    >>> velocity = coeffs_derivative(coeffs, order=1)
    >>> print(velocity)
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (3,)   ┆ float32 ┆ [  0.      ,   0.      , -31.4...
    ...
    """
    if order < 0:
        raise ValueError("order must be a non-negative integer.")
    if isinstance(X, FourierCoeffs):
        if wd is not None:
            raise ValueError("Pass either FourierCoeffs or raw coeffs with wd, not both.")
        wd = _frequency_to_wd(X.frequency)
    elif wd is None:
        raise ValueError("wd must be provided when differentiating raw coefficient arrays.")
    coeffs = coeffs_to_complex(X)
    harmonic_ids = jnp.arange(coeffs.size)
    dcoeffs = (1j * harmonic_ids * jnp.asarray(wd)) ** order * coeffs
    if isinstance(X, FourierCoeffs):
        return complex_to_coeffs(dcoeffs, frequency=X.frequency)
    return complex_to_coeffs(dcoeffs)


def coeffs_to_table(X: ArrayLike | FourierCoeffs) -> jax.Array:
    """Return a JAX table with stacked real and imaginary phasor parts."""
    complex_coeffs = coeffs_to_complex(X)
    return jnp.stack([jnp.real(complex_coeffs), jnp.imag(complex_coeffs)], axis=0)


def time_grid(wd: ArrayLike, n: int, oversample: int = 1) -> jax.Array:
    """Build an evenly spaced grid over one forcing period."""
    _require_positive_factor("n", n)
    _require_positive_factor("oversample", oversample)
    period = 2.0 * jnp.pi / jnp.asarray(wd)
    sample_count = n * oversample
    return jnp.linspace(0.0, period, sample_count, endpoint=False)


def coeffs_to_time_signal(
    X: ArrayLike | FourierCoeffs, oversample: int = 1
) -> jax.Array | SampledSignal:
    """Synthesize a real time signal from stacked real Fourier coefficients.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = FourierCoeffs(values=jnp.array([0.0, 1.0, 0.0]), frequency=5.0)
    >>> signal = coeffs_to_time_signal(coeffs, oversample=4)
    >>> print(signal)
    SampledSignal
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (8,)   ┆ float32 ┆ [1.        , 0.70710677, 0.   ...
    ...
    """
    _require_positive_factor("oversample", oversample)
    complex_coeffs = coeffs_to_complex(X)
    harmonic_count = complex_coeffs.size - 1
    sample_count = max(2 * harmonic_count * oversample, 1)

    # irfft expects positive-frequency amplitudes scaled by the FFT convention.
    fft_coeffs = complex_coeffs * (sample_count / 2.0)
    fft_coeffs = fft_coeffs.at[0].set(complex_coeffs[0] * sample_count)
    values = jnp.fft.irfft(fft_coeffs, n=sample_count).real
    if isinstance(X, FourierCoeffs):
        frequency = jnp.asarray(X.frequency)
        if jnp.any(frequency == 0):
            raise ValueError(
                "FourierCoeffs.frequency must be non-zero to build a sampled signal."
            )
        return SampledSignal(values=values, frequency=frequency)
    return values


def time_to_coeffs(
    x: ArrayLike | SampledSignal, downsample: int = 1
) -> jax.Array | FourierCoeffs:
    """Project a sampled real time signal back onto the HBM basis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> signal = SampledSignal(
    ...     values=jnp.cos(2.0 * jnp.pi * jnp.arange(16) / 16),
    ...     frequency=5.0,
    ... )
    >>> coeffs = time_to_coeffs(signal, downsample=4)
    >>> print(coeffs)
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (5,)   ┆ float32 ┆ [ 4.0854182e-08,  1.0000000e+0...
    ...
    """
    _require_positive_factor("downsample", downsample)
    signal = _sample_values(x)
    sample_count = signal.size
    if sample_count != 1 and sample_count % (2 * downsample) != 0:
        raise ValueError(
            "Signal length must be 1 or divisible by 2 * downsample."
        )

    fft_coeffs = jnp.fft.rfft(signal) * 2.0 / sample_count
    fft_coeffs = fft_coeffs.at[0].set(fft_coeffs[0] * 0.5)
    kept_coeffs = fft_coeffs[: sample_count // (2 * downsample) + 1]
    if isinstance(x, SampledSignal):
        return complex_to_coeffs(kept_coeffs, frequency=jnp.asarray(x.frequency))
    return complex_to_coeffs(kept_coeffs)


def coeffs_pow(
    X: ArrayLike | FourierCoeffs, p: ArrayLike, oversample: int = 1
) -> jax.Array | FourierCoeffs:
    """Raise a periodic signal to a power and re-project it onto the HBM basis."""
    signal = coeffs_to_time_signal(X, oversample=oversample)
    if isinstance(signal, SampledSignal):
        return time_to_coeffs(
            SampledSignal(values=signal.values ** p, frequency=signal.frequency),
            downsample=oversample,
        )
    return time_to_coeffs(signal ** p, downsample=oversample)
