from . import tracing
from . import inductor_passthrough
from . import null
from . import keepaway

backend_dict = {
        "inductor_passthrough": inductor_passthrough.backend,
        "tracing": tracing.backend,
        "null": null.backend,
        "keepaway": keepaway.backend
}

