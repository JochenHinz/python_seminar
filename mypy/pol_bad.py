from typing import Sequence, Self
from itertools import zip_longest
from more_itertools import convolve


class Polynomial:

    def _deriv(self) -> 'Polynomial':
        return Polynomial([a * i for i, a in enumerate(self.weights[1:], 1)])

    def __init__(self, weights: Sequence[float | int]) -> None:
        self.weights = weights

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
        if n == 0: return self
        return self._deriv().derivative(n-1)

    def __add__(self, other: Self) -> 'Polynomial':
        return Polynomial(list(map(sum, zip_longest(self.weights, other.weights, fillvalue=0))))

    def __mul__(self, other: Self) -> 'Polynomial':
        return Polynomial(list(convolve(self.weights, other.weights)))

    # This time out of place
    def raise_order(self) -> 'Polynomial':
        return Polynomial([0] + self.weights)


def main():
    pol = Polynomial((1, 2, 3))
    # raise order
    pol.raise_order()
    # fractional derivative
    pol.derivative(1.5)
    # Only ints
    pol = Polynomial([1, 2, 3])
    pol(1)


if __name__ == '__main__':
    main()