from typing import Sequence, Self, Tuple
from itertools import zip_longest, dropwhile
from more_itertools import convolve


def truncate_trailing_zeros(tpl: Sequence[int|float]) -> Tuple[float, ...]:
    """
        [] => ()
        [1, 0, 0] => (1.0,)
        (0, 1, 0) => (0.0, 1.0)
    """
    return tuple(dropwhile(lambda x: x==0, map(float, reversed(tpl))))[::-1]


class Polynomial:

    def _deriv(self) -> 'Polynomial':
        return Polynomial([a * i for i, a in enumerate(self.weights[1:], 1)])

    def __init__(self, weights: Sequence[float | int]) -> None:
        # convert to tuple of floats and truncate trailing zeros, if any.
        # if passed Sequence type is empty, coerce to (0.0,) (zero polynomial)
        self.weights = truncate_trailing_zeros(weights) or (0.0,)

    @property
    def order(self) -> int:
        return len(self.weights) - 1

    def __call__(self, x: int | float) -> float:
        return sum(a * x**i for i, a in enumerate(self.weights))

    def __repr__(self) -> str:
        ret = ' + '.join(reversed([f'{a}x^{i}' for i, a in enumerate(self.weights) if a]))\
                   .replace('x^0', '') \
                   .replace(' 1x', ' x')\
                   .replace('+ -', '- ')
        if not ret: return '0'
        return ret

    def derivative(self, n: int = 1) -> 'Polynomial':
        # make sure n converted to an int is positive
        assert (n := int(n)) >= 0
        if n == 0: return self
        return self._deriv().derivative(n-1)

    def __add__(self, other: Self) -> 'Polynomial':
        return Polynomial(list(map(sum, zip_longest(self.weights, other.weights, fillvalue=0))))

    def __mul__(self, other: Self) -> 'Polynomial':
        return Polynomial(list(convolve(self.weights, other.weights)))

    def raise_order(self) -> 'Polynomial':
        # prepend (0,) as a tuple.
        # this will always work because self.weights is always a tuple.
        return Polynomial((0,) + self.weights)


def main():
    pol = Polynomial((1, 2, 3))
    # raise order
    pol.raise_order()
    # fractional derivative
    pol.derivative(1.5)


if __name__ == '__main__':
    main()
