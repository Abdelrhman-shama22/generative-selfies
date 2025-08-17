import os, gc, random
import numpy as np
import pandas as pd
import selfies as sf
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ================== Paths =================
DATA_PATH = r"C:\abdo123\gdb9\final_dataset_with_selfies.csv"
OUTPUT_SELFIES_PATH = r"C:\abdo123\gdb9\new_selfies.csv"

# ================== Hyperparams =================
BATCH_SIZE = 64
EPOCHS = 60
MAX_LEN_LIMIT = 200
EMBED_DIM = 128
ENC_UNITS = 256
DEC_UNITS = 256
LATENT_DIM = 64
VAL_FRAC = 0.1
LR = 5e-4

# ================== GPU Config =================
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

# ================== Load Data =================
df = pd.read_csv(DATA_PATH)
selfies_list = df['selfies'].astype(str).tolist()

seqs = [list(sf.split_selfies(s)) for s in selfies_list]
max_len = min(MAX_LEN_LIMIT, max(len(s) for s in seqs))

special = ['[nop]', '[SOS]', '[EOS]']
alphabet = sorted(list({tok for s in seqs for tok in s} | set(special)))
stoi = {t:i for i,t in enumerate(alphabet)}
itos = {i:t for t,i in stoi.items()}
vocab_size = len(alphabet)

def encode_sequence(tokens, max_len):
    toks = tokens[:max_len-2]
    seq = ['[SOS]'] + toks + ['[EOS]']
    if len(seq) < max_len:
        seq += ['[nop]'] * (max_len - len(seq))
    return [stoi.get(t, stoi['[nop]']) for t in seq]

encoded = np.array([encode_sequence(s, max_len) for s in seqs], dtype=np.int32)

decoder_inputs  = encoded.copy()
targets         = encoded.copy()
decoder_inputs[:, 1:] = encoded[:, :-1]
decoder_inputs[:, 0]  = stoi['[SOS]']

# ================== Split =================
N = len(encoded)
val_n = max(1, int(N * VAL_FRAC))
train_idx = np.arange(0, N - val_n)
val_idx   = np.arange(N - val_n, N)

enc_in_train, dec_in_train, y_train = encoded[train_idx], decoder_inputs[train_idx], targets[train_idx]
enc_in_val, dec_in_val, y_val = encoded[val_idx], decoder_inputs[val_idx], targets[val_idx]

train_ds = tf.data.Dataset.from_tensor_slices(((enc_in_train, dec_in_train), y_train))\
    .shuffle(len(train_idx)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(((enc_in_val, dec_in_val), y_val))\
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================== Model: Encoder =================
enc_in = Input(shape=(max_len,), dtype=tf.int32, name='enc_in')
emb_enc = Embedding(vocab_size, EMBED_DIM, name='emb_enc')(enc_in)
emb_enc = Dropout(0.3)(emb_enc)

enc_bi = Bidirectional(LSTM(ENC_UNITS, return_sequences=False, return_state=True,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-5)), name='bi_lstm')
enc_out, f_h, f_c, b_h, b_c = enc_bi(emb_enc)
enc_state_h = tf.concat([f_h, b_h], axis=-1)
enc_state_c = tf.concat([f_c, b_c], axis=-1)

z_mean = Dense(LATENT_DIM, name='z_mean')(enc_state_h)
z_log_var = Dense(LATENT_DIM, name='z_log_var')(enc_state_h)
z_log_var = Lambda(lambda t: tf.clip_by_value(t, -10.0, 10.0), name='clip')(z_log_var)

def sampling(args):
    zm, zv = args
    eps = tf.random.normal(shape=(tf.shape(zm)[0], LATENT_DIM), dtype=tf.float32)
    return zm + tf.exp(0.5 * zv) * eps

z = Lambda(sampling, name='sampling')([z_mean, z_log_var])

h0 = Dense(DEC_UNITS, activation='tanh', name='h0')(z)
c0 = Dense(DEC_UNITS, activation='tanh', name='c0')(z)

# ================== Model: Decoder =================
dec_in = Input(shape=(max_len,), dtype=tf.int32, name='dec_in')
emb_dec = Embedding(vocab_size, EMBED_DIM, name='emb_dec')(dec_in)
emb_dec = Dropout(0.3)(emb_dec)

dec_rnn = GRU(DEC_UNITS, return_sequences=True,
              kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='gru_dec')
dec_out = dec_rnn(emb_dec, initial_state=h0)
dec_out = Dropout(0.3)(dec_out)
logits = TimeDistributed(Dense(vocab_size, activation='softmax'), name='td_out')(dec_out)

decoder = Model([dec_in, z], logits, name='decoder')

# ================== VAE end-to-end =================
outputs = decoder([dec_in, z])
vae = Model([enc_in, dec_in], outputs, name='vae')

# ================== Loss & Compile =================
kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)
kl = -0.5 * tf.reduce_mean(1 + tf.cast(z_log_var, tf.float32) - tf.square(z_mean) - tf.exp(z_log_var))
vae.add_loss(kl_weight * kl)

opt = Adam(learning_rate=LR, clipnorm=1.0)
vae.compile(optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

class KLWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10): self.warmup = warmup_epochs
    def on_epoch_begin(self, epoch, logs=None):
        kl_weight.assign(float(min(1.0, (epoch + 1) / self.warmup)))
    def on_epoch_end(self, epoch, logs=None): gc.collect()

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=6, restore_best_weights=True
)
plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

history = vae.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[KLWarmup(10), early, plateau],
    verbose=1
)

# ================== Sampling Function =================
def sample_from_probs(probs, top_k=30, top_p=0.9, repetition_penalty=1.2, prev_tokens=None):
    probs = np.asarray(probs).astype(np.float64)

    if prev_tokens is not None:
        for t in set(prev_tokens):
            probs[t] /= repetition_penalty

    if top_k > 0:
        idx = np.argpartition(-probs, top_k)[:top_k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[idx] = True
        probs = np.where(mask, probs, 0)

    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cum_probs = np.cumsum(sorted_probs) / np.sum(sorted_probs)
    cutoff = cum_probs <= top_p
    cutoff[np.argmax(cum_probs > top_p)] = True
    mask = np.zeros_like(probs, dtype=bool)
    mask[sorted_idx[cutoff]] = True
    probs = np.where(mask, probs, 0)

    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)

# ================== Generate SELFIES =================
def generate_selfies(n_samples=200, max_len=max_len):
    samples = []
    for _ in range(n_samples):
        z_sample = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        dec_in_seed = np.zeros((1, max_len), dtype=np.int32)
        dec_in_seed[0, 0] = stoi['[SOS]']
        prev_tokens = []
        seq = []
        for t in range(1, max_len):
            preds = decoder.predict([dec_in_seed, z_sample], verbose=0)
            probs = preds[0, t-1]
            next_tok = sample_from_probs(probs, top_k=30, top_p=0.9,
                                         repetition_penalty=1.2, prev_tokens=prev_tokens)
            tok = itos[next_tok]
            if tok == '[EOS]':
                break
            if tok not in ['[SOS]', '[nop]']:
                seq.append(tok)
            dec_in_seed[0, t] = next_tok
            prev_tokens.append(next_tok)
        selfies_str = ''.join(seq)
        samples.append(selfies_str)
    return samples

gen_selfies = generate_selfies(200, max_len)
pd.DataFrame({"generated_selfies": gen_selfies}).to_csv(OUTPUT_SELFIES_PATH, index=False)
print(f"✅ New generated selfies saved to {OUTPUT_SELFIES_PATH}")


