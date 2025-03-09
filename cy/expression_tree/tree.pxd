#cython: language_level=3

# [HELPER] - enum tokenizing various mathematical functions
cdef enum ExpressionType:
  CONSTANT,
  VARIABLE,
  EXP,
  SIN,
  COS,
  ADD,
  MULTIPLY,
  POWER


# [HELPER] - Structs representing the different mathematical operations


# f(x) = value
cdef struct Constant:
  double value


# f(x0, ..., xn) = xi;  token: 0 for `x0` 1 for `x1`...
cdef struct Variable:
  int token


# Addition, multiplication, ... between several args
cdef struct ArithmeticOperation:
  int nargs              # number of arguments
  Expression* arguments  # pointer to first position of array of arguments


# f(x)^g(x)
cdef struct Power:
  Expression* base
  Expression* exponent


# exp(f(x)), sin(f(x)), cos(f(x)), ...
cdef struct Function:
  Expression* argument


# [HELPER] - Expression union combining all math structs
cdef union ExpressionUnion:
  Constant constant
  Variable variable
  ArithmeticOperation arithmeticOperation
  Function function
  Power power


# Struct holding a token of the expression type
# and the corresponding expression as an ExpressionUnion

# Example:

# type: ExpressionType.EXP
# expression: ExpressionUnion with expression.function = struct Function(*argument)

# [HELPER] - Expression struct combining the type and the expression
cdef struct Expression:
  ExpressionType type
  ExpressionUnion expression


# [SETUP] - Forward definitions of select functions defined in the main script.
cdef Expression parse_math_string(str math_string)
cdef void eval_expression_vectorized(Expression expr, double[:, ::1] arg, double* into) except *
cdef void free_Expression(Expression* expression)
