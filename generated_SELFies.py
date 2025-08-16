import os, gc, numpy as np, pandas as pd, selfies as sf, tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


DATA_PATH = r"C:\abdo123\gdb9\final_dataset_with_selfies.csv"
OUTPUT_SELFIES_PATH = r"C:\abdo123\gdb9\generated_selfies.csv"

BATCH_SIZE = 64
EPOCHS = 60
MAX_LEN_LIMIT = 200
EMBED_DIM = 128
ENC_UNITS = 256
DEC_UNITS = 256
LATENT_DIM = 64
VAL_FRAC = 0.1
LR = 5e-4

#  GPU 
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass


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

#  Split 
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
emb_enc = Dropout(0.3)(emb_enc)   #  dropout added

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
emb_dec = Dropout(0.3)(emb_dec)   #  dropout added

dec_rnn = GRU(DEC_UNITS, return_sequences=True, 
              kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='gru_dec')

dec_out = dec_rnn(emb_dec, initial_state=h0)
dec_out = Dropout(0.3)(dec_out)   #  dropout added
logits = TimeDistributed(Dense(vocab_size, activation='softmax'), name='td_out')(dec_out)

decoder = Model([dec_in, z], logits, name='decoder')

# ================== VAE end-to-end =================
outputs = decoder([dec_in, z])
vae = Model([enc_in, dec_in], outputs, name='vae')

# ================== Losses & Compile ===============
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

# ================== Save generated selfies =========
def generate_selfies(n_samples=100):
    samples = []
    for _ in range(n_samples):
        z_sample = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        dec_in_seed = np.zeros((1, max_len), dtype=np.int32)
        dec_in_seed[0,0] = stoi['[SOS]']
        preds = decoder.predict([dec_in_seed, z_sample], verbose=0)
        seq = [itos[np.argmax(p)] for p in preds[0]]
        selfies_str = ''.join([t for t in seq if t not in ['[SOS]','[EOS]','[nop]']])
        samples.append(selfies_str)
    return samples

gen_selfies = generate_selfies(200)
pd.DataFrame({"generated_selfies": gen_selfies}).to_csv(OUTPUT_SELFIES_PATH, index=False)
print(f" Generated selfies saved to {OUTPUT_SELFIES_PATH}")
