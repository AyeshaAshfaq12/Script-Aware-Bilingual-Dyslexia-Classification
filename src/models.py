"""Phase 4: backbone, SCA module, and all experimental arms.

Guide section 5. Every arm takes the same two inputs, (image,
script_id), and returns a single sigmoid output, so the training loop
and the data pipeline are identical across arms. Arms that ignore the
script simply do not read the second input.

The matched-capacity design (paper section 4.4, guide section 5.2):
A1 and A2 are the SAME architecture with one flag flipped. The
unconditioned branch still contains and builds the script embedding
table, so trainable-parameter counts are identical by construction.
`assert_capacity_match` enforces this and is called in the smoke tests.

Script id encoding is fixed in src/data.py: urdu=0, english=1, digit=2.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from tensorflow.keras import layers

from data import IMG_SIZE, N_SCRIPTS

# Insertion depths, guide section 5.1. Verified present in TF 2.20.
TAPS = {
    "early": "conv_pw_3_relu",
    "mid": "conv_pw_7_relu",
    "late": "conv_pw_13_relu",
}
DEPTH_CONFIGS = ["early", "mid", "late", "all"]


def taps_for(depth_config: str) -> list[str]:
    if depth_config == "all":
        return [TAPS[k] for k in ("early", "mid", "late")]
    if depth_config not in TAPS:
        raise ValueError(f"depth_config must be one of {DEPTH_CONFIGS}")
    return [TAPS[depth_config]]


# --------------------------------------------------------------------
# SCA
# --------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="sad")
class SCA(layers.Layer):
    """Script-conditioned channel attention, squeeze-and-excitation style.

    conditioned=True  -> gates see the script embedding      (arm A2)
    conditioned=False -> gates see a learned constant vector  (arm A1)

    Both settings instantiate the embedding table and the constant, so
    parameter counts match exactly. The unconditioned branch calls the
    embedding (keeping the graph shape identical) and discards it.
    """

    def __init__(self, channels: int, d: int = 16, r: int = 16,
                 conditioned: bool = True, **kw):
        super().__init__(**kw)
        self.channels = int(channels)
        self.d = int(d)
        self.r = int(r)
        self.conditioned = bool(conditioned)
        self.embed = layers.Embedding(N_SCRIPTS, self.d, name="script_embed")
        self.fc1 = layers.Dense(max(self.channels // self.r, 4),
                                activation="relu", name="sca_fc1")
        self.fc2 = layers.Dense(self.channels, activation="sigmoid",
                                name="sca_fc2")
        self.const = self.add_weight(
            name="const_embed", shape=(self.d,),
            initializer="zeros", trainable=True)

    def build(self, f_shape, script_id_shape=None):
        self.embed.build(script_id_shape or ())
        self.fc1.build((f_shape[0], self.channels + self.d))
        self.fc2.build((f_shape[0], self.fc1.units))
        super().build(f_shape)

    def compute_output_shape(self, f_shape, script_id_shape=None):
        return f_shape

    def call(self, f, script_id):
        z = tf.reduce_mean(f, axis=[1, 2])            # squeeze -> (B, C)
        e_cond = self.embed(script_id)                # always built
        if self.conditioned:
            e = e_cond
        else:
            e = tf.tile(self.const[None, :], [tf.shape(f)[0], 1])
        a = self.fc2(self.fc1(tf.concat([z, e], axis=-1)))
        return f + f * a[:, None, None, :]            # residual gate

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "d": self.d, "r": self.r,
                    "conditioned": self.conditioned})
        return cfg


@keras.saving.register_keras_serializable(package="sad")
class ScriptOneHot(layers.Layer):
    """One-hot encode the script id. A registered layer rather than a
    Lambda, so saved models reload without unsafe deserialisation."""

    def __init__(self, depth: int = N_SCRIPTS, **kw):
        super().__init__(**kw)
        self.depth = int(depth)

    def call(self, script_id):
        return tf.one_hot(script_id, self.depth)

    def compute_output_shape(self, script_id_shape):
        return (script_id_shape[0], self.depth)

    def get_config(self):
        cfg = super().get_config()
        cfg["depth"] = self.depth
        return cfg


@keras.saving.register_keras_serializable(package="sad")
class RouteSelect(layers.Layer):
    """Select one of K head outputs with a one-hot routing vector."""

    def call(self, heads, onehot):
        return tf.reduce_sum(heads * onehot, axis=-1, keepdims=True)

    def compute_output_shape(self, heads_shape, onehot_shape):
        return (heads_shape[0], 1)


# --------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------
def _mobilenet_v1(weights: str | None = "imagenet") -> keras.Model:
    base = keras.applications.MobileNet(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights=weights)
    base.trainable = False                            # freeze all conv layers
    return base


def _assert_linear_chain(base: keras.Model) -> None:
    """MobileNetV1 must be a single chain for layer-by-layer re-hosting."""
    for lyr in base.layers[1:]:
        n_in = len(lyr._inbound_nodes)
        assert n_in == 1, f"{lyr.name} has {n_in} inbound nodes"


def _head(x, dropout: float = 0.5):
    """The source study's head (guide section 5.3, arm A0)."""
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    return layers.Dropout(dropout, name="dropout")(x)


def _inputs():
    img = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    sid = layers.Input((), dtype="int32", name="script_id")
    return img, sid


# --------------------------------------------------------------------
# arms
# --------------------------------------------------------------------
def build_a0(backbone: str = "mobilenet", weights: str | None = "imagenet",
             dropout: float = 0.5, name: str | None = None) -> keras.Model:
    """A0: re-run baselines. Frozen backbone + the source study's head.

    The script input is accepted and ignored, so every arm shares one
    data pipeline.
    """
    img, sid = _inputs()
    shape = (IMG_SIZE, IMG_SIZE, 3)
    ctor = {
        "mobilenet": keras.applications.MobileNet,
        "mobilenetv2": keras.applications.MobileNetV2,
        "mobilenetv3small": keras.applications.MobileNetV3Small,
        "vgg16": keras.applications.VGG16,
        "inceptionv3": keras.applications.InceptionV3,
    }
    if backbone == "cnn_scratch":
        x = img
        for i, f in enumerate([32, 64, 128]):
            x = layers.Conv2D(f, 3, padding="same", activation="relu",
                              name=f"cnn_conv{i}")(x)
            x = layers.MaxPooling2D(2, name=f"cnn_pool{i}")(x)
    else:
        if backbone not in ctor:
            raise ValueError(f"unknown backbone '{backbone}'")
        base = ctor[backbone](input_shape=shape, include_top=False,
                              weights=weights)
        base.trainable = False
        x = base(img, training=False)

    x = _head(x, dropout)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    return keras.Model([img, sid], out, name=name or f"A0_{backbone}")


def _build_sca_arm(conditioned: bool, depth_config: str = "mid",
                   d: int = 16, r: int = 16, dropout: float = 0.5,
                   weights: str | None = "imagenet",
                   name: str | None = None) -> keras.Model:
    """Shared builder for A1 (conditioned=False) and A2 (conditioned=True)."""
    base = _mobilenet_v1(weights)
    _assert_linear_chain(base)
    tap_names = set(taps_for(depth_config))

    img, sid = _inputs()
    x = img
    seen = set()
    for lyr in base.layers[1:]:
        x = lyr(x)
        if lyr.name in tap_names:
            x = SCA(x.shape[-1], d=d, r=r, conditioned=conditioned,
                    name=f"sca_{lyr.name}")(x, sid)
            seen.add(lyr.name)
    missing = tap_names - seen
    assert not missing, f"tap layers not found in backbone: {missing}"

    x = _head(x, dropout)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    default = f"{'A2' if conditioned else 'A1'}_{depth_config}_d{d}_r{r}"
    return keras.Model([img, sid], out, name=name or default)


def build_a1(**kw) -> keras.Model:
    """A1: matched-capacity script-AGNOSTIC control."""
    return _build_sca_arm(conditioned=False, **kw)


def build_a2(**kw) -> keras.Model:
    """A2: script-CONDITIONED model (ours). Oracle script id."""
    return _build_sca_arm(conditioned=True, **kw)


def build_a3(dropout: float = 0.5,
             weights: str | None = "imagenet") -> keras.Model:
    """A3: naive conditioning. One-hot script concatenated onto the
    128-dim penultimate features. No SCA module."""
    base = _mobilenet_v1(weights)
    img, sid = _inputs()
    x = base(img, training=False)
    x = _head(x, dropout)
    onehot = ScriptOneHot(name="script_onehot")(sid)
    x = layers.Concatenate(name="concat_script")([x, onehot])
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    return keras.Model([img, sid], out, name="A3_onehot_concat")


def build_script_classifier(weights: str | None = "imagenet") -> keras.Model:
    """Lightweight script classifier: frozen features + Dense(3, softmax).

    Used to supply predicted script ids for A2' and A5. Trained on the
    same train partition; its val/test accuracy is reported.
    """
    base = _mobilenet_v1(weights)
    img = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = base(img, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(N_SCRIPTS, activation="softmax", name="script_out")(x)
    return keras.Model(img, out, name="script_classifier")


def build_a5(dropout: float = 0.5,
             weights: str | None = "imagenet") -> keras.Model:
    """A5: identify-then-branch. Shared frozen features route to one of
    three script-specific Dense(128)+Dense(1) heads.

    The script_id input carries the PREDICTED script (from
    build_script_classifier), matching the two-stage pipeline the guide
    specifies. Routing is a one-hot select over the three head outputs.
    """
    base = _mobilenet_v1(weights)
    img, sid = _inputs()
    feat = layers.Flatten(name="flatten")(base(img, training=False))

    outs = []
    for k, tag in enumerate(("urdu", "english", "digit")):
        h = layers.Dense(128, activation="relu", name=f"dense_128_{tag}")(feat)
        h = layers.Dropout(dropout, name=f"dropout_{tag}")(h)
        outs.append(layers.Dense(1, activation="sigmoid",
                                 name=f"out_{tag}")(h))
    stacked = layers.Concatenate(name="stack_heads")(outs)   # (B, 3)
    onehot = ScriptOneHot(name="route_onehot")(sid)
    out = RouteSelect(name="route_select")(stacked, onehot)
    return keras.Model([img, sid], out, name="A5_identify_then_branch")


# A4 (per-script experts) is not a single Keras model: it is two
# independently trained A0 models plus a routing rule. Built in train.py.
def build_a4_expert(dropout: float = 0.5,
                    weights: str | None = "imagenet",
                    tag: str = "expert") -> keras.Model:
    """One A4 expert: the A0 MobileNet arm trained on a single-script
    subset. Routing across experts happens at inference."""
    return build_a0("mobilenet", weights=weights, dropout=dropout,
                    name=f"A4_{tag}")


# --------------------------------------------------------------------
# capacity check (mandatory, guide section 5.3)
# --------------------------------------------------------------------
def param_counts(model: keras.Model) -> dict[str, int]:
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return {"total": trainable + non_trainable,
            "trainable": trainable,
            "non_trainable": non_trainable}


def assert_capacity_match(cfg: dict) -> dict:
    """assert count_params(A1) == count_params(A2) for one config."""
    a1, a2 = build_a1(**cfg), build_a2(**cfg)
    p1, p2 = param_counts(a1), param_counts(a2)
    assert p1 == p2, (f"capacity mismatch at {cfg}: A1={p1} A2={p2}")
    return p1


if __name__ == "__main__":
    print(f"{'depth':6s} {'d':>3s} {'r':>3s} "
          f"{'total':>12s} {'trainable':>12s}  match")
    for depth in DEPTH_CONFIGS:
        for d in (8, 16):
            for r in (8, 16):
                cfg = {"depth_config": depth, "d": d, "r": r,
                       "weights": None}
                p = assert_capacity_match(cfg)
                print(f"{depth:6s} {d:3d} {r:3d} "
                      f"{p['total']:12,d} {p['trainable']:12,d}  OK")


# --------------------------------------------------------------------
# split builders: frozen cacheable prefix + trainable suffix
# --------------------------------------------------------------------
# The backbone is frozen and no augmentation is used (hard rule 3), so
# the output of every frozen layer is a constant function of the input
# image. Computing that prefix once per depth config and training only
# the suffix is EXACTLY equivalent to end-to-end training, and is what
# makes the local-CPU budget feasible. Equivalence is asserted in
# tests/test_smoke.py::TestSplitEquivalence.

_A0_CTORS = {
    "mobilenet": "MobileNet",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3small": "MobileNetV3Small",
    "vgg16": "VGG16",
    "inceptionv3": "InceptionV3",
}


def _frozen_backbone(backbone: str, weights: str | None):
    base = getattr(keras.applications, _A0_CTORS[backbone])(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights=weights)
    base.trainable = False
    return base


def build_split(arm: str, backbone: str = "mobilenet",
                depth_config: str = "mid", d: int = 16, r: int = 16,
                dropout: float = 0.5, weights: str | None = "imagenet"):
    """Return (prefix, suffix, full).

    prefix : image -> frozen features            (None if nothing is frozen)
    suffix : (features, script_id) -> output     (the trainable part)
    full   : (image, script_id) -> output        (shares suffix weights)

    arm in {A0, A1, A2, A3, A5, script_clf}. A2' reuses A2 with predicted
    script ids; A4 reuses A0 per script subset.
    """
    img, sid = _inputs()

    # ---- arms with a fully frozen backbone -------------------------
    if arm in ("A0", "A3", "A5", "script_clf"):
        if backbone == "cnn_scratch":
            if arm != "A0":
                raise ValueError("cnn_scratch is only used for A0")
            return None, None, build_a0("cnn_scratch", weights=None,
                                        dropout=dropout)
        base = _frozen_backbone(backbone, weights)
        prefix = keras.Model(img, base(img, training=False), name="prefix")

        fin = layers.Input(prefix.output_shape[1:], name="features")
        if arm == "script_clf":
            z = layers.GlobalAveragePooling2D(name="gap")(fin)
            sout = layers.Dense(N_SCRIPTS, activation="softmax",
                                name="script_out")(z)
            suffix = keras.Model(fin, sout, name="suffix")
            full = keras.Model(img, suffix(prefix(img)),
                               name="script_classifier")
            return prefix, suffix, full

        fsid = layers.Input((), dtype="int32", name="script_id")
        if arm == "A5":
            flat = layers.Flatten(name="flatten")(fin)
            outs = []
            for tag in ("urdu", "english", "digit"):
                h = layers.Dense(128, activation="relu",
                                 name=f"dense_128_{tag}")(flat)
                h = layers.Dropout(dropout, name=f"dropout_{tag}")(h)
                outs.append(layers.Dense(1, activation="sigmoid",
                                         name=f"out_{tag}")(h))
            stacked = layers.Concatenate(name="stack_heads")(outs)
            onehot = ScriptOneHot(name="route_onehot")(fsid)
            sout = RouteSelect(name="route_select")(stacked, onehot)
        else:
            z = _head(fin, dropout)
            if arm == "A3":
                onehot = ScriptOneHot(name="script_onehot")(fsid)
                z = layers.Concatenate(name="concat_script")([z, onehot])
            sout = layers.Dense(1, activation="sigmoid", name="out")(z)

        suffix = keras.Model([fin, fsid], sout, name="suffix")
        full = keras.Model([img, sid], suffix([prefix(img), sid]),
                           name=f"{arm}_{backbone}")
        return prefix, suffix, full

    # ---- SCA arms: prefix ends at the FIRST tap --------------------
    if arm not in ("A1", "A2"):
        raise ValueError(f"unknown arm '{arm}'")
    conditioned = (arm == "A2")

    base = _mobilenet_v1(weights)
    _assert_linear_chain(base)
    tap_names = set(taps_for(depth_config))
    chain = base.layers[1:]
    names = [l.name for l in chain]
    first = min(names.index(t) for t in tap_names)

    px = img
    for lyr in chain[:first + 1]:
        px = lyr(px)
    prefix = keras.Model(img, px, name="prefix")

    fin = layers.Input(prefix.output_shape[1:], name="features")
    fsid = layers.Input((), dtype="int32", name="script_id")
    sx = SCA(fin.shape[-1], d=d, r=r, conditioned=conditioned,
             name=f"sca_{names[first]}")(fin, fsid)
    for lyr in chain[first + 1:]:
        sx = lyr(sx)
        if lyr.name in tap_names:
            sx = SCA(sx.shape[-1], d=d, r=r, conditioned=conditioned,
                     name=f"sca_{lyr.name}")(sx, fsid)
    sx = _head(sx, dropout)
    sout = layers.Dense(1, activation="sigmoid", name="out")(sx)

    suffix = keras.Model([fin, fsid], sout, name="suffix")
    full = keras.Model([img, sid], suffix([prefix(img), sid]),
                       name=f"{arm}_{depth_config}_d{d}_r{r}")
    return prefix, suffix, full
