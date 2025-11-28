from datetime import datetime
import inspect
from functools import wraps
from molml_mcp.config import LOG_PATH


def loggable(func):
    """
    Decorator that logs:
      - function name
      - first line of docstring
      - ORIGINAL inputs (before mutation)
      - full return value (outputs)

    Writes to `server.log` in a human-readable, multiline format.
    """

    def _fmt(val):
        """Compact representation to avoid huge log lines."""
        try:
            # Truncate long text fields
            if isinstance(val, str):
                if len(val) > 100:
                    return val[:97] + "..."
                return val
            
            # Represent array-like objects compactly
            if hasattr(val, "shape"):  # e.g. pandas DataFrame / numpy array
                return f"<Array-like shape={val.shape}>"
            
            # Represent large collections compactly
            if isinstance(val, (list, dict, tuple, set)) and len(val) > 30:
                return f"<{type(val).__name__} len={len(val)}>"
            return repr(val)
        except Exception:
            return "<unprintable>"

    sig = inspect.signature(func)
    doc = inspect.getdoc(func)


    @wraps(func)
    def wrapper(*args, **kwargs):
        # Capture ORIGINAL inputs as passed into the function
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        original_inputs = dict(bound.arguments)

        # Time the function execution
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        # Prepare docstring line
        if doc:
            docstring_first_line = doc.strip().split("\n")[0]
        else:
            docstring_first_line = "Description not available."

        # Format inputs and outputs
        inputs_str = {k: _fmt(v) for k, v in original_inputs.items()}

        if isinstance(result, dict):
            outputs_str = {k: _fmt(v) for k, v in result.items()}
        else:
            outputs_str = _fmt(result)

        # Build log entry in your original style
        entry = (
            datetime.now().strftime("\n%Y-%m-%d %H:%M:%S:\n")
            + f"\tFunction: {func.__name__}()\n"
            + f"\tDescription: {docstring_first_line}\n"
            + f"\tInputs: {inputs_str}\n"
            + f"\tOutputs: {outputs_str}\n"
            + f"\tExecution Time: {elapsed_time:.4f}s\n"
        )

        # Write to log file
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(entry)

        return result

    return wrapper