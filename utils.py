"""Utilities used by example notebooks"""

import functools


class lazyproperty:
    """This decorator works exactly like the @property decorator, but it
    caches the return value, so subsequent calls with the same
    calling parameters are fast.

    This is useful in when running the method is an expensive or slow
    operation, such as when making database calls.

    However it was originally conceived of to support a
    functional-programming style.  In particular, to support creation
    of a "pure function". See
    https://en.wikipedia.org/wiki/Pure_function. Here, the pure
    function is implemented with access to a Python object store which
    caches values.

    The above in greater detail...

    @lazyproperty is like @property. When used to decorate methods
    having only a `self` parameter, it is accessed like an attribute
    on an instance, i.e. trailing parentheses are not used.

    Unlike @property, the decorated method is only evaluated on first
    access; the resulting value is cached and that same value returned
    on second and later access without re-evaluation of the method.

    Like @property, this class produces a *data descriptor* object, which is
    stored in the __dict__ of the *class* under the name of the decorated
    method ('fget' nominally). The cached value is stored in the __dict__ of
    the *instance* under that same name.

    Because it is a data descriptor (as opposed to a *non-data descriptor*),
    its `__get__()` method is executed on each access of the decorated
    attribute; the __dict__ item of the same name is "shadowed" by the
    descriptor.

    While this may represent a performance improvement over a property, its
    greater benefit may be its other characteristics. One common use is to
    construct collaborator objects, removing that "real work" from the
    constructor, while still only executing once. It also de-couples client
    code from any sequencing considerations; if it's accessed from more than
    one location, it's assured it will be ready whenever needed.

    Loosely based on: https://stackoverflow.com/a/6849299/1902513.

    A lazyproperty is read-only. There is no counterpart to the optional
    "setter" (or deleter) behavior of an @property. This is critically
    important to maintaining its immutability and idempotence guarantees.
    Attempting to assign to a lazyproperty raises AttributeError
    unconditionally.

    The parameter names in the methods below correspond to this usage
    example::

        class Obj(object)

            @lazyproperty
            def fget(self):
                return 'some result'

        obj = Obj()

    Not suitable for wrapping a function (as opposed to a method) because it
    is not callable.

    """

    def __init__(self, fget):
        """*fget* is the decorated method (a "getter" function).

        A lazyproperty is read-only, so there is only an *fget* function (a
        regular @property can also have an fset and fdel function). This name
        was chosen for consistency with Python's `property` class which uses
        this name for the corresponding parameter.
        """
        # ---maintain a reference to the wrapped getter method
        self._fget = fget
        # ---adopt fget's __name__, __doc__, and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, type=None):
        """Called on each access of 'fget' attribute on class or instance.

        *self* is this instance of a lazyproperty descriptor "wrapping" the
        property method it decorates (`fget`, nominally).

        *obj* is the "host" object instance when the attribute is accessed
        from an object instance, e.g. `obj = Obj(); obj.fget`. *obj* is None
        when accessed on the class, e.g. `Obj.fget`.

        *type* is the class hosting the decorated getter method (`fget`) on
        both class and instance attribute access.
        """
        # ---when accessed on class, e.g. Obj.fget, just return this
        # ---descriptor instance (patched above to look like fget).
        if obj is None:
            return self

        # ---when accessed on instance, start by checking instance __dict__
        value = obj.__dict__.get(self.__name__)
        if value is None:
            # ---on first access, __dict__ item will absent. Evaluate fget()
            # ---and store that value in the (otherwise unused) host-object
            # ---__dict__ value of same name ('fget' nominally)
            value = self._fget(obj)
            obj.__dict__[self.__name__] = value
        return value

    def __set__(self, obj, value):
        """Raises unconditionally, to preserve read-only behavior.

        This decorator is intended to implement immutable (and idempotent)
        object attributes. For that reason, assignment to this property must
        be explicitly prevented.

        If this __set__ method was not present, this descriptor would become
        a *non-data descriptor*. That would be nice because the cached value
        would be accessed directly once set (__dict__ attrs have precedence
        over non-data descriptors on instance attribute lookup). The problem
        is, there would be nothing to stop assignment to the cached value,
        which would overwrite the result of `fget()` and break both the
        immutability and idempotence guarantees of this decorator.

        The performance with this __set__() method in place was roughly 0.4
        usec per access when measured on a 2.8GHz development machine; so
        quite snappy and probably not a rich target for optimization efforts.
        """
        raise AttributeError("can't set attribute")


def preprocess_ms_image(x):
    # define mean and std values
    mean = [
        1353.036,
        1116.468,
        1041.475,
        945.344,
        1198.498,
        2004.878,
        2376.699,
        2303.738,  # 8
        732.957,
        12.092,
        1818.820,
        1116.271,
        2602.579,  # 8A
    ]
    std = [
        65.479,
        154.008,
        187.997,
        278.508,
        228.122,
        356.598,
        456.035,
        531.570,  # 8
        98.947,
        1.188,
        378.993,
        303.851,
        503.181,  # 8A
    ]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x
