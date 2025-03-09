# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

""" [IMPORT] """
cimport cython
from cython.parallel cimport prange
from cython.view cimport array
from sympy import parse_expr

from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp as cexp, sin as csin, cos as ccos

import numpy as np
cimport numpy as cnp

import time


""" Expression constructors and destructor """


""" [HELPER] - various expression constructors """


cdef Expression make_ArithmeticOperation(ExpressionType type, Expression* args, int nargs):
  """
    Construct an :Expression: ret with
        ret.type = type (must be ADD or MULTIPLY)
        ret.expression.arithmeticOperation.arguments = args and
        ret.expression.arithmeticOperation.nargs = nargs.
  """
  cdef:
    Expression ret
    ExpressionUnion expr
    ArithmeticOperation arop

  if not type in (ADD, MULTIPLY):
    abort()

  ret.type = type
  arop.arguments = args
  arop.nargs = nargs
  expr.arithmeticOperation = arop
  ret.expression = expr

  return ret


cdef Expression make_Constant(double value):
  """
    Construct an :Expression: ret with
    ret.type = CONSTANT
    ret.expression.constant = Constant(value)
  """
  cdef:
    Expression ret
    ExpressionUnion expr

  expr.constant = Constant(value)
  ret.type = CONSTANT
  ret.expression = expr

  return ret


cdef Expression make_Variable(int token):
  """
    Construct an :Expression: ret with
    ret.type = VARIABLE
    ret.expression.variable.token = token
  """
  cdef:
    Expression ret
    ExpressionUnion expr

  assert token >= 0

  expr.variable = Variable(token)
  ret.type = VARIABLE
  ret.expression = expr

  return ret


cdef Expression make_Function(ExpressionType type, Expression* arg):
  """
    Construct an :Expression: ret with
    ret.type = type, where type is either EXP, SIN or COS and
    ret.expression.function.argument = arg
  """
  cdef:
    Expression ret
    ExpressionUnion expr
    Function func

  if not type in (EXP, SIN, COS):
    abort()

  ret.type = type
  func.argument = arg
  expr.function = func
  ret.expression = expr

  return ret


cdef Expression make_Exponent(Expression* base, Expression* exponent):
  """
    Construct an :Expression: ret with
    ret.type = POWER
    ret.expression.power.base = base
    ret.expression.power.exponent = exponent
  """
  cdef:
    Expression ret
    ExpressionUnion expr

  expr.power = Power(base, exponent)
  ret.type = POWER
  ret.expression = expr

  return ret


""" [HELPER] - deallocate a variable of type `Expression` """
cdef void free_Expression(Expression* expression):
  cdef:
    int i

  if expression.type in (ADD, MULTIPLY):

    for i in range(expression.expression.arithmeticOperation.nargs):
      free_Expression(&expression.expression.arithmeticOperation.arguments[i])
    free(expression.expression.arithmeticOperation.arguments)

  elif expression.type in (EXP, SIN, COS):

    free_Expression(&expression.expression.function.argument[0])
    free(expression.expression.function.argument)

  elif expression.type == POWER:

    free_Expression(&expression.expression.power.base[0])
    free_Expression(&expression.expression.power.exponent[0])
    free(expression.expression.power.base)
    free(expression.expression.power.exponent)

  else:
    pass



""" MATHEMATICAL FUNCTION EVALUATION """


""" [FOCUS] - Evaluate a mathematical expression """
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double eval_expression(Expression expr, double[:] args) nogil:
  """
    Evaluate a :struct: Expression representing a mathematical function.

    Parameters
    ----------
    expr: :struct: Expression
      The struct instantiation representing the mathematical function.
    args: :struct: double[:]
      The 1D memoryview representing the coordinate to evaluate in.
      If args.shape[0] does not equal the number of arguments in `expr`,
      a segfault may occur.
  """
  cdef:
    double ret
    int i

  if expr.type == CONSTANT:
    return expr.expression.constant.value
  elif expr.type == VARIABLE:
    return args[expr.expression.variable.token]
  elif expr.type == ADD:
    ret = 0.0
    for i in range(expr.expression.arithmeticOperation.nargs):
      ret += eval_expression(expr.expression.arithmeticOperation.arguments[i], args)
    return ret
  elif expr.type == MULTIPLY:
    ret = 1.0
    for i in range(expr.expression.arithmeticOperation.nargs):
      ret *= eval_expression(expr.expression.arithmeticOperation.arguments[i], args)
    return ret
  elif expr.type == POWER:
    return eval_expression(expr.expression.power.base[0], args) ** \
           eval_expression(expr.expression.power.exponent[0], args)
  elif expr.type == EXP:
    return cexp(eval_expression(expr.expression.function.argument[0], args))
  elif expr.type == SIN:
    return csin(eval_expression(expr.expression.function.argument[0], args))
  elif expr.type == COS:
    return ccos(eval_expression(expr.expression.function.argument[0], args))
  else:
    abort()


""" [FOCUS] - Evaluate a mathematical expression vectorized (and parallelized) """
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void eval_expression_vectorized(Expression expr, double[:, ::1] arg, double* into) except *:
  """
    Evaluate a :struct: Expression representing a mathematical function in an array of coordinates.

    Parameters
    ----------
    expr: :struct: Expression
      The struct instantiation representing the mathematical function.
    args: :struct: double[:, ::1]
      The 2D memoryview representing the coordinates to evaluate in.
      If args.shape[1] does not equal the number of arguments in `expr`,
      a segfault may occur.
    into: double*
      A pointer to the first entry of the preallocated array to iterate the
      result into.
  """
  cdef int i
  with nogil:
    for i in prange(arg.shape[0]):
      into[i] = eval_expression(expr, arg[i])



""" MATH STRING PARSING """


""" [HELPER] - use sympy to decompose a string into a list of lists of mathematical operations """
def decompose_expression(expr):
  """
    Given a sympy expression `expr` create a list of lists representing
    the sympy expression using primitive types (strings, floats, ...)
  """
  cdef str name = expr.func.__name__
  if name in ('Add', 'Mul'):
    args = expr.args
    decomposed_args = [decompose_expression(arg) for arg in args]
    return [name, decomposed_args]
  elif name == 'Pow':
    args = expr.args
    return [name, [decompose_expression(arg) for arg in args]]
  elif name in ('exp', 'sin', 'cos'):
    arg, = expr.args
    return [name, [decompose_expression(arg)]]
  else:
    try:
      return [float(str(expr)), []]
    except Exception:
      ret = str(expr)
      assert ret.startswith('x')
      assert int(ret[1:]) >= 0
      return [ret, []]


""" [HELPER] - Convert a list of lists into a Cython expression """
cdef Expression expression_from_list_of_lists(list lol):
  """
    Convert a list of lists comprised of primitives (strings, floats, ...)
    into an instance of `Expression`, to be used in `eval_expression_vectorized`.
  """
  cdef:
    int nargs
    Expression* base
    Expression* exponent
    Expression* args

  head, tail = lol
  nargs = len(tail)

  cdef Expression ret

  if head in ('Mul', 'Add'):

    assert nargs >= 2
    args = <Expression*>malloc(nargs * sizeof(Expression))
    for i in range(nargs):
      args[i] = expression_from_list_of_lists(tail[i])
    return make_ArithmeticOperation({'Add': ADD, 'Mul': MULTIPLY}[head], &args[0], nargs)

  elif head in ('exp', 'sin', 'cos'):

    assert nargs == 1
    args = <Expression*>malloc(1 * sizeof(Expression))
    args[0] = expression_from_list_of_lists(tail[0])
    return make_Function({'exp': EXP, 'sin': SIN, 'cos': COS}[head], &args[0])

  elif head == 'Pow':

    assert nargs == 2
    base = <Expression*>malloc(sizeof(Expression))
    exponent = <Expression*>malloc(sizeof(Expression))
    base[0] = expression_from_list_of_lists(tail[0])
    exponent[0] = expression_from_list_of_lists(tail[1])
    return make_Exponent(&base[0], &exponent[0])

  else:

    if isinstance(head, str):
      assert head.startswith('x')
      return make_Variable(int(head[1:]))

    return make_Constant(float(head))



""" [FOCUS] - use 1) `decompose_expression` and
                  2) `expression_from_list_of_lists`
                  to create a Cython expression from a math string """
cdef Expression parse_math_string(str math_string):

  # convert math string to sympy object
  sympy_expression = parse_expr(math_string)

  # decompost sympy expression into a tree of mathematical operations
  decomposed_expression = decompose_expression(sympy_expression)

  # create a cython expression representing the tree
  expr = expression_from_list_of_lists(decomposed_expression)

  return expr



""" [FOCUS] - Function to evaluate a mathematical string, callable from pure Python """
def evaluate_math_string(str math_string, x):
  """
    Evaluate a function by its mathematical string representation.
    The function can take several arguments, which are denoted by
    x0, x1, ...
    The position `x` must have shape (N, n), where `n` is the
    number of arguments in the string.

    >>> x = np.array([[1.0, 0], [0.0, 1.0]])
    >>> math_string = 'x0 - x1'
    >>> evaluate_math_string(math_string, x)
      np.array([1.0, -1.0])
  """
  cdef:
    double[::1] into
    Expression expr

  # allocate flat memory for return array
  into = np.empty((np.prod(x.shape[:-1]),), dtype=np.float64)

  # convert math string to cython expression
  expr = parse_math_string(math_string)

  # evaluate cython expression in `x` reshaped to shape (N, n) where `n`
  # is the number of variables. Iterate the result into `into`.
  eval_expression_vectorized(expr, x.reshape(-1, x.shape[-1]), &into[0])

  # deallocate cython expression
  free_Expression(&expr)

  # convert to numpy array, reshape to original shape and return
  return np.asarray(into).reshape(x.shape[:-1])
