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
import polars as pl
from jax.typing import ArrayLike

from .core import namedtuple_repr

__all__ = [
    "SampledSignal",
    "fourier_zeros",
    "cosine_forcing",
    "sine_forcing",
    "phased_forcing",
    "format_fourier_coeffs",
    "print_fourier_coeffs",
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


def fourier_zeros(order: int, dtype: ArrayLike = 0.0) -> jax.Array:
    """Build a zero coefficient vector up to harmonic ``order``."""
    if order < 0:
        raise ValueError("order must be >= 0.")
    return jnp.zeros(2 * order + 1, dtype=jnp.result_type(jnp.asarray(dtype)))


def cosine_forcing(amplitude: ArrayLike, harmonic: int, order: int) -> jax.Array:
    """Build ``amplitude * cos(harmonic * wd * t)`` in stacked real form."""
    if order < 0:
        raise ValueError("order must be >= 0.")
    if harmonic < 1 or harmonic > order:
        raise ValueError("harmonic must satisfy 1 <= harmonic <= order.")
    coeffs = fourier_zeros(order=order, dtype=amplitude)
    return coeffs.at[harmonic].set(jnp.asarray(amplitude))


def sine_forcing(amplitude: ArrayLike, harmonic: int, order: int) -> jax.Array:
    """Build ``amplitude * sin(harmonic * wd * t)`` in stacked real form."""
    if order < 0:
        raise ValueError("order must be >= 0.")
    if harmonic < 1 or harmonic > order:
        raise ValueError("harmonic must satisfy 1 <= harmonic <= order.")
    coeffs = fourier_zeros(order=order, dtype=amplitude)
    return coeffs.at[order + harmonic].set(jnp.asarray(amplitude))


def phased_forcing(
    amplitude: ArrayLike,
    phase: ArrayLike,
    harmonic: int,
    order: int,
) -> jax.Array:
    """Build ``amplitude * cos(harmonic * wd * t + phase)``."""
    amplitude = jnp.asarray(amplitude)
    phase = jnp.asarray(phase)
    return cosine_forcing(
        amplitude=amplitude * jnp.cos(phase),
        harmonic=harmonic,
        order=order,
    ) - sine_forcing(
        amplitude=amplitude * jnp.sin(phase),
        harmonic=harmonic,
        order=order,
    )


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


def _coeff_values(X: ArrayLike) -> jax.Array:
    """Return the raw coefficient vector."""
    return jnp.asarray(X)


def _sample_values(x: ArrayLike | SampledSignal) -> jax.Array:
    """Return the raw sampled signal from an array or a SampledSignal object."""
    if isinstance(x, SampledSignal):
        return jnp.asarray(x.values)
    return jnp.asarray(x)


def _frequency_to_wd(frequency: ArrayLike) -> jax.Array:
    """Convert a frequency in Hz to an angular frequency in rad/s."""
    return 2.0 * jnp.pi * jnp.asarray(frequency)


def _coeffs_table_frame(X: ArrayLike) -> pl.DataFrame:
    """Return a printable table for a stacked real coefficient vector."""
    table = coeffs_to_table(X)
    columns = {"basis": ["cos", "sin"]}
    for harmonic in range(table.shape[1]):
        columns[str(harmonic)] = table[:, harmonic].tolist()
    return pl.DataFrame(columns)


def format_fourier_coeffs(X: ArrayLike) -> str:
    """Return a compact table repr for a stacked real coefficient vector."""
    with pl.Config(tbl_width_chars=160, fmt_str_lengths=48):
        table = _coeffs_table_frame(X)
    return f"FourierCoeffs\n{table}"


def print_fourier_coeffs(X: ArrayLike) -> None:
    """Print Fourier coefficients with harmonics as columns and basis rows."""
    print(format_fourier_coeffs(X))


class SampledSignal(NamedTuple):
    """Uniformly sampled signal over one period, tagged by its frequency in Hz.

    The arithmetic implemented on this container is limited to the linear
    operations that preserve the sampling layout directly:

    - addition and subtraction between sampled signals
    - unary plus and minus
    - multiplication and division by scalars

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


def add_fourier_coeffs(a: ArrayLike, b: ArrayLike) -> jax.Array:
    """Add two Fourier coefficient vectors."""
    return jnp.asarray(a) + jnp.asarray(b)


def sub_fourier_coeffs(a: ArrayLike, b: ArrayLike) -> jax.Array:
    """Subtract two Fourier coefficient vectors."""
    return jnp.asarray(a) - jnp.asarray(b)


def scale_fourier_coeffs(X: ArrayLike, scalar: ArrayLike) -> jax.Array:
    """Multiply a Fourier coefficient vector by a scalar."""
    return jnp.asarray(X) * jnp.asarray(scalar)


def sum_fourier_coeffs(*terms: ArrayLike) -> jax.Array:
    """Sum multiple Fourier coefficient vectors."""
    if len(terms) == 0:
        raise ValueError("At least one Fourier coefficient vector is required.")
    return sum((jnp.asarray(term) for term in terms[1:]), start=jnp.asarray(terms[0]))


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


def coeffs_to_complex(X: ArrayLike) -> jax.Array:
    """Convert stacked real Fourier coefficients to positive-frequency phasors."""
    coeffs = _coeff_values(X)
    N = _harmonic_order_count(coeffs)
    a0 = coeffs[0]
    a = coeffs[1 : N + 1]
    b = coeffs[N + 1 :]
    return jnp.concatenate([a0[None], a - 1j * b])


def complex_to_coeffs(C: ArrayLike) -> jax.Array:
    """Convert positive-frequency phasors back to stacked real coefficients."""
    complex_coeffs = jnp.asarray(C)
    if complex_coeffs.size < 1:
        raise ValueError("At least one complex coefficient is required.")
    a0 = jnp.real(complex_coeffs[0])
    a = jnp.real(complex_coeffs[1:])
    b = -jnp.imag(complex_coeffs[1:])
    real_dtype = jnp.real(complex_coeffs).dtype
    return jnp.concatenate([a0[None], a, b]).astype(real_dtype)


def coeffs_derivative(X: ArrayLike, wd: ArrayLike, order: int = 1) -> jax.Array:
    """Differentiate a coefficient vector with respect to time.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = jnp.array([0.0, 1.0, 0.0])
    >>> velocity = coeffs_derivative(coeffs, wd=2.0 * jnp.pi * 5.0, order=1)
    >>> velocity
    Array([  0.      ,   0.      , -31.415928], dtype=float32)
    """
    if order < 0:
        raise ValueError("order must be a non-negative integer.")
    coeffs = coeffs_to_complex(X)
    harmonic_ids = jnp.arange(coeffs.size)
    dcoeffs = (1j * harmonic_ids * jnp.asarray(wd)) ** order * coeffs
    return complex_to_coeffs(dcoeffs)


def coeffs_to_table(X: ArrayLike) -> jax.Array:
    """Return a 2 x (N + 1) table with cosine and sine coefficients."""
    coeffs = _coeff_values(X)
    N = _harmonic_order_count(coeffs)
    cosine = coeffs[: N + 1]
    sine = jnp.concatenate([jnp.zeros_like(coeffs[:1]), coeffs[N + 1 :]])
    return jnp.stack([cosine, sine], axis=0)


def time_grid(wd: ArrayLike, n: int, oversample: int = 1) -> jax.Array:
    """Build an evenly spaced grid over one forcing period."""
    _require_positive_factor("n", n)
    _require_positive_factor("oversample", oversample)
    period = 2.0 * jnp.pi / jnp.asarray(wd)
    sample_count = n * oversample
    return jnp.linspace(0.0, period, sample_count, endpoint=False)


def coeffs_to_time_signal(
    X: ArrayLike,
    frequency: ArrayLike | None = None,
    oversample: int = 1,
) -> jax.Array | SampledSignal:
    """Synthesize a real time signal from stacked real Fourier coefficients.

    When ``frequency`` is provided in Hz, the result is wrapped as a
    ``SampledSignal`` so the time grid is available directly.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = jnp.array([0.0, 1.0, 0.0])
    >>> signal = coeffs_to_time_signal(coeffs, frequency=5.0, oversample=4)
    >>> signal.values
    Array([ 1.0000000e+00,  7.0710677e-01, -4.3711388e-08, -7.0710677e-01,
           -1.0000000e+00, -7.0710677e-01,  1.1924881e-08,  7.0710677e-01],      dtype=float32)
    """
    _require_positive_factor("oversample", oversample)
    complex_coeffs = coeffs_to_complex(X)
    harmonic_count = complex_coeffs.size - 1
    sample_count = max(2 * harmonic_count * oversample, 1)

    # irfft expects positive-frequency amplitudes scaled by the FFT convention.
    fft_coeffs = complex_coeffs * (sample_count / 2.0)
    fft_coeffs = fft_coeffs.at[0].set(complex_coeffs[0] * sample_count)
    values = jnp.fft.irfft(fft_coeffs, n=sample_count).real
    if frequency is None:
        return values

    frequency = jnp.asarray(frequency)
    try:
        if bool(jnp.any(frequency == 0)):
            raise ValueError("frequency must be non-zero to build a sampled signal.")
    except jax.errors.TracerBoolConversionError:
        pass
    return SampledSignal(values=values, frequency=frequency)


def time_to_coeffs(x: ArrayLike | SampledSignal, downsample: int = 1) -> jax.Array:
    """Project a sampled real time signal back onto the HBM basis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> signal = SampledSignal(
    ...     values=jnp.cos(2.0 * jnp.pi * jnp.arange(16) / 16),
    ...     frequency=5.0,
    ... )
    >>> coeffs = time_to_coeffs(signal, downsample=4)
    >>> coeffs
    Array([ 4.0854182e-08,  1.0000000e+00, -3.7318731e-08,  0.0000000e+00,
            0.0000000e+00], dtype=float32)
    """
    _require_positive_factor("downsample", downsample)
    signal = _sample_values(x)
    sample_count = signal.size
    if sample_count != 1 and sample_count % (2 * downsample) != 0:
        raise ValueError("Signal length must be 1 or divisible by 2 * downsample.")

    fft_coeffs = jnp.fft.rfft(signal) * 2.0 / sample_count
    fft_coeffs = fft_coeffs.at[0].set(fft_coeffs[0] * 0.5)
    kept_coeffs = fft_coeffs[: sample_count // (2 * downsample) + 1]
    return complex_to_coeffs(kept_coeffs)


def coeffs_pow(X: ArrayLike, p: ArrayLike, oversample: int = 1) -> jax.Array:
    """Raise a periodic signal to a power and re-project it onto the HBM basis."""
    signal = coeffs_to_time_signal(X, oversample=oversample)
    return time_to_coeffs(signal ** p, downsample=oversample)
