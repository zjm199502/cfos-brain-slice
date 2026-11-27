"""Channel model selection, saved model names and cell segmentation."""
from . import cellpose4

DEFAULT_MODEL = cellpose4.MODEL_KEY
NAMES = {DEFAULT_MODEL: "Cellpose 4（cpsam_v2）"}


def default_model(channel_name):
    """Return the default for a new channel."""
    return DEFAULT_MODEL


def normalize_model_id(model):
    """Map retired project settings to the current default before new inference."""
    key = str(model or "").strip().lower()
    if key in ("", "stardist", "2d_versatile_fluo") or key.startswith("stardist:"):
        return DEFAULT_MODEL
    return model


def recorded_model_name(key):
    """Display the original model recorded in a completed result."""
    if key is None:
        return "模型未记录"
    if str(key).strip().lower() in ("", "stardist", "2d_versatile_fluo", "stardist:2d_versatile_fluo"):
        return "StarDist（荧光细胞核）"
    return NAMES.get(key, str(key))


def predict(image, params, cancel=None, progress=None):
    if cancel and cancel():
        raise InterruptedError
    labels = cellpose4.predict_labels(
        image, cellprob_threshold=params.cellprob_threshold, cancel=cancel, progress=progress
    )
    return labels, cellpose4.backend_text()
