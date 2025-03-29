from types import EllipsisType
from copy import copy
from itertools import chain
from typing import Hashable, Tuple, Union, Any, Iterable, Optional, Callable
from numpy.typing import NDArray
from numpy import tensordot, transpose, isclose, log, exp
from numpy.linalg import norm
from ordered_set import OrderedSet  # type: ignore


def _check_unique(*indices_ids: Hashable) -> None:
    seen = set(indices_ids)
    if len(seen) < len(indices_ids):
        raise ValueError(f"Index IDs in {indices_ids} are not unique")


def _check_ids_number(
    array: NDArray,
    *index_ids: Hashable,
) -> None:
    if len(index_ids) != len(array.shape):
        raise ValueError(
            f"Number of indices in NXArray array is {len(array.shape)} must be equal to the number of index IDs which is {len(index_ids)}"
        )


def _common_ids(
    lhs: Iterable[Hashable],
    rhs: Iterable[Hashable],
) -> OrderedSet[Hashable]:
    return OrderedSet(lhs).intersection(OrderedSet(rhs))


def _not_common_ids(
    lhs: Iterable[Hashable],
    rhs: Iterable[Hashable],
) -> OrderedSet[Hashable]:
    return OrderedSet(lhs).symmetric_difference(OrderedSet(rhs))


class NXArray:
    """
    Array with named indices.

    Args:
        array: raw array;
        indices_ids: sequence of names of each index.

    Raises:
        ValueError: raises if IDs in `index_ids` are not unique;
        ValueError: raises if rank of the raw array does not match number of IDs given.
    """

    def __init__(
        self,
        array: NDArray,
        *index_ids: Hashable,
        **kwargs: Any,
    ) -> None:
        _check_unique(*index_ids)
        _check_ids_number(array, *index_ids)
        _norm = norm(array)
        self._log_norm = log(_norm)
        if kwargs.get("add_log_norm"):
            self._log_norm += kwargs["log_norm"]
        self._array = array / _norm
        self._index_ids = index_ids

    """Gives sequense of index IDs present in the NXArray, order of IDs is no specified."""

    @property
    def index_ids(self) -> Tuple[Hashable, ...]:
        return self._index_ids

    """Gives shape of the underlying raw array which matches sequence of index IDs obtained by `.index_ids`."""

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._array.shape

    """Gives rank (number of indices) of the NXArray."""

    @property
    def rank(self) -> int:
        return len(self.shape)

    """Returns natural logorithm of the 2-norm."""

    @property
    def log_norm(self) -> NDArray:
        return self._log_norm

    """Returns 2-norm"""

    @property
    def norm(self) -> NDArray:
        return exp(self._log_norm)

    def _release_transposed_immutable_view(
        self, *index_ids: Hashable
    ) -> NDArray:
        assert self.rank == len(index_ids)
        if len(self._array.shape) == 0:
            return self._array
        else:
            result = self._transpose(*index_ids)._array.view()
            result.flags.writeable = False
            return result

    """
    Returns a raw normalized array (|arr|_2 = 1).

    Args:
        indices_ids: sequence of index IDs specifying order of indices in the output array.

    Returns:
        raw array.

    Raises:
        ValueError: raises when number of index IDs does not match the rank of an array;
        ValueError: raises when index IDs are not unique;
        ValueError: raises when some ID form `index_ids` is not present.
    """

    def release_normalized_array(self, *index_ids: Hashable) -> NDArray:
        if len(index_ids) == self.rank:
            return self._release_transposed_immutable_view(*index_ids)
        else:
            raise ValueError(
                f"Number of index IDs must macth the rank of an array {self.rank}, got {len(index_ids)} indices IDs"
            )

    """
    Returns a raw array.

    Args:
        indices_ids: sequence of index IDs specifying order of indices in the output array.

    Returns:
        raw array.

    Raises:
        ValueError: raises when number of index IDs does not match the rank of an array;
        ValueError: raises when index IDs are not unique;
        ValueError: raises when some ID form `index_ids` is not present.
    """

    def release_array(self, *index_ids: Hashable) -> NDArray:
        return self.release_normalized_array(*index_ids) * self.norm

    """
    Relabels indices according to the passed closure.
    """

    def relabel(self, func: Callable[[Hashable], Hashable]) -> None:
        self._index_ids = tuple(map(func, self._index_ids))

    def _transpose(self, *new_subsystems_order: Hashable) -> "NXArray":
        assert len(self.index_ids) == len(
            new_subsystems_order
        ), f"{len(self.index_ids)}, {len(new_subsystems_order)}"
        new_raw_order = map(self.index_ids.index, new_subsystems_order)
        new_state_arr = transpose(self._array, tuple(new_raw_order))
        new_nxarr = NXArray(
            new_state_arr, *new_subsystems_order, add_log_norm=self.log_norm
        )
        return new_nxarr

    def _back_partial_transpose(
        self, *new_subsystems_order: Hashable
    ) -> "NXArray":
        _check_unique(new_subsystems_order)
        rest_ids = OrderedSet(self.index_ids).difference(new_subsystems_order)
        new_subsystems_order_completed = tuple(
            chain(rest_ids, new_subsystems_order)
        )
        return self._transpose(*new_subsystems_order_completed)

    def _front_partial_transpose(
        self, *new_subsystems_order: Hashable
    ) -> "NXArray":
        _check_unique(new_subsystems_order)
        rest_ids = OrderedSet(self.index_ids).difference(new_subsystems_order)
        new_subsystems_order_completed = tuple(
            chain(new_subsystems_order, rest_ids)
        )
        return self._transpose(*new_subsystems_order_completed)

    def _partial_transpose(
        self,
        *new_subsystems_order: Union[EllipsisType, Hashable],
    ) -> "NXArray":
        if not new_subsystems_order:
            return self
        elif new_subsystems_order[0] is Ellipsis:
            return self._back_partial_transpose(*new_subsystems_order[1:])
        else:
            return self._front_partial_transpose(*new_subsystems_order)

    """
    Contract two arrays over common index IDs or multiply an array by a given scalar.

    Args:
        other: second NXArray of a scalar.

    Returns:
        resulting NXArray.
    """

    def __mul__(self, other: Union["NXArray", float]) -> "NXArray":
        if isinstance(other, NXArray):
            common_ids = _common_ids(self.index_ids, other.index_ids)
            common_size = len(common_ids)
            new_arr = tensordot(
                self._partial_transpose(..., *common_ids)._array,
                other._partial_transpose(*common_ids)._array,
                common_size,
            )
            rest_ids = _not_common_ids(self.index_ids, other.index_ids)
            result_nxarr = NXArray(
                new_arr, *rest_ids, add_log_norm=self.log_norm + other.log_norm
            )
            return result_nxarr
        elif isinstance(other, float):
            result_nxarr = copy(self)
            result_nxarr._log_norm += log(other)
            return result_nxarr

    def __rmul__(self, other: float) -> "NXArray":
        if isinstance(other, float):
            return self.__mul__(other)
        else:
            return NotImplemented

    def _getitem(
        self,
        idx: Tuple[Union[EllipsisType, Hashable], ...],
    ) -> Tuple[int, "NXArray"]:
        new_nxarr = self._partial_transpose(*idx)
        if idx[0] is Ellipsis:
            matrix_barrier = self.rank - len(idx) + 1
        else:
            matrix_barrier = len(idx)
        return (matrix_barrier, new_nxarr)

    """Transforms NXArray to feed it into matrix decomposition subroutines.

    Args:
        idx: index IDs and optional ... (Ellipsis) which tell how to split indices into two groups to form a matrix.
            Some examples: if NXArray has the following index IDs `"i1", "i2", "i3", "i4"` then
            by calling arr["i2", "i4"] one splits indices into the following groups `("i2", "i4")` and `("i1", "i3")`,
            if one calls arr[..., "i2", "i4"] then order is reversed, i.e. `("i1", "i3")` and `("i2", "i4")`

    Returns:
        tuple of updated NXArray and int number telling a matrix decomposition subroutines how to act.

    Raises:
        ValueError: raises when index IDs are not unique;
        ValueError: raises when some ID form `index_ids` is not present.
    """

    def __getitem__(
        self,
        idx: Union[
            Tuple[Union[EllipsisType, Hashable], ...],
            Union[EllipsisType, Hashable],
        ],
    ) -> Tuple[int, "NXArray"]:
        if not isinstance(idx, tuple):
            return self._getitem((idx,))
        else:
            return self._getitem(idx)

    def _other_to_same_order(self, other: "NXArray") -> Optional["NXArray"]:
        if self.rank != other.rank:
            return None
        else:
            try:
                return other._transpose(*self.index_ids)
            except ValueError:
                return None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NXArray):
            ordered_other = self._other_to_same_order(other)
            if ordered_other is None:
                return False
            else:
                return (
                    self.shape == ordered_other.shape
                    and bool(isclose(self._array, ordered_other._array).all())
                    and bool(isclose(self._log_norm, ordered_other._log_norm))
                )
        return NotImplemented

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(index_ids: {self.index_ids}, shape: {self.shape})"
