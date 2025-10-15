# diag_bench_tf.py (patched)
import os, time, platform, json, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

def print_header():
    print("=== System ===")
    print("OS:", platform.platform())
    print("Python:", platform.python_version())
    print("TensorFlow:", tf.__version__)
    print("\n=== TF Devices ===")
    for d in tf.config.list_physical_devices():
        print(" -", d)

def build_conv1d(seq_len=4000, n_classes=3):
    inp = tf.keras.layers.Input(shape=(seq_len, 1))
    x = inp
    for f in [64, 128, 256]:
        x = tf.keras.layers.Conv1D(f, 5, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy")
    return model

def make_synth_ds(batch_size=64, seq_len=4000, steps=200, n_classes=3):
    x = np.random.randint(0, 2, size=(batch_size*steps, seq_len, 1), dtype=np.uint8)
    y = np.random.randint(0, n_classes, size=(batch_size*steps,), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)) \
        .batch(batch_size) \
        .map(lambda a,b: (tf.cast(a, tf.float32), b)) \
        .prefetch(tf.data.AUTOTUNE)
    return ds

def bench(device_name, batch_size=64, seq_len=4000, steps=200, n_classes=3, warmup=20):
    model = build_conv1d(seq_len, n_classes)
    ds = make_synth_ds(batch_size, seq_len, steps, n_classes)

    # Warm-up
    it = iter(ds)
    for _ in range(warmup):
        xb, yb = next(it)
        with tf.device(device_name):
            model.train_on_batch(xb, yb)   # <— zonder reset_metrics kwarg

    # Timed
    t0 = time.time()
    cnt = 0
    model.reset_metrics()  # optioneel; voor consistentie
    for xb, yb in ds:
        with tf.device(device_name):
            model.train_on_batch(xb, yb)   # <— idem
        cnt += 1
    dt = time.time() - t0
    sps = (batch_size * cnt) / dt
    return {"device": device_name, "batches": cnt, "batch_size": batch_size,
            "seq_len": seq_len, "steps_per_sec": cnt/dt,
            "samples_per_sec": sps, "time_s": dt}

def main():
    print_header()
    print("\n=== Benchmark ===")
    seq_len, batch, steps = 4000, 64, 200
    results = []
    results.append(bench("/CPU:0", batch, seq_len, steps))
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        try:
            results.append(bench("/GPU:0", batch, seq_len, steps))
        except Exception as e:
            print("GPU bench failed:", e)
    print("\nResults:")
    print(json.dumps(results, indent=2))
    if len(results) > 1:
        r_cpu = results[0]["samples_per_sec"]
        r_gpu = results[1]["samples_per_sec"]
        print(f"\niGPU speedup vs CPU: x{(r_gpu/r_cpu):.2f}")

if __name__ == "__main__":
    main()
