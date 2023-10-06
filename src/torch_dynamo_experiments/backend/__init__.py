from . import tracing
from . import inductor_passthrough
from . import null
from . import keepaway
from . import interpreter
from . import recompile
from functorch.compile import debug_compile

backend_dict = {
    "inductor_passthrough": inductor_passthrough.backend,
    "tracing": tracing.backend,
    "null": null.backend,
    "keepaway": keepaway.backend,
    "inductor": "inductor",
    "interpreter": interpreter.backend,
    "recompile": recompile.backend,
    "debug_compile": debug_compile
}
