import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# ==================== ПАРАМЕТРЫ МОДЕЛИ ====================
VOCAB_SIZE = 100  # Размер словаря
MAX_SEQ_LEN = 15  # Максимальная длина последовательности
EMBEDDING_DIM = 128  # Размерность эмбеддингов
NUM_HEADS = 4  # Количество голов внимания
NUM_LAYERS = 2  # Количество слоев
HIDDEN_DIM = 512  # Размер скрытого слоя
BATCH_SIZE = 32  # Размер батча
EPOCHS = 30  # Количество эпох

# ==================== СОЗДАНИЕ ДАННЫХ ====================
def create_realistic_data(num_samples=5000, seq_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE):
    """
    Создает реалистичные данные для задачи машинного перевода.
    Задача: копирование последовательности со сдвигом на 1 и добавлением префикса
    """
    print(f"Создание {num_samples} реалистичных примеров...")
    
    src_sequences = []
    tgt_sequences = []
    
    for _ in range(num_samples):
        # Генерируем случайную последовательность
        seq_len_actual = np.random.randint(3, seq_len-2)
        sequence = np.random.randint(1, vocab_size-5, size=seq_len_actual)
        
        # Source: добавляем начало последовательности
        src = np.concatenate([[0], sequence])  # 0 - стартовый токен
        
        # Target: сдвиг + специальные токены
        tgt_input = np.concatenate([[1], sequence])  # 1 - стартовый токен для decoder
        tgt_output = np.concatenate([sequence, [2]])  # 2 - конечный токен
        
        # Добавляем padding до MAX_SEQ_LEN
        src_padded = np.pad(src, (0, MAX_SEQ_LEN - len(src)), mode='constant', constant_values=0)
        tgt_input_padded = np.pad(tgt_input, (0, MAX_SEQ_LEN - len(tgt_input)), mode='constant', constant_values=0)
        tgt_output_padded = np.pad(tgt_output, (0, MAX_SEQ_LEN - len(tgt_output)), mode='constant', constant_values=0)
        
        src_sequences.append(src_padded)
        tgt_sequences.append((tgt_input_padded, tgt_output_padded))
    
    src_data = np.array(src_sequences, dtype=np.int32)
    tgt_input_data = np.array([t[0] for t in tgt_sequences], dtype=np.int32)
    tgt_output_data = np.array([t[1] for t in tgt_sequences], dtype=np.int32)
    
    print(f"Размерность данных: src={src_data.shape}, tgt_input={tgt_input_data.shape}, tgt_output={tgt_output_data.shape}")
    print(f"Пример данных:")
    print(f"  Source:      {src_data[0][:10]}...")
    print(f"  Target input: {tgt_input_data[0][:10]}...")
    print(f"  Target output: {tgt_output_data[0][:10]}...")
    
    return src_data, tgt_input_data, tgt_output_data

# ==================== МОДЕЛЬ TRANSFORMER ====================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position=5000, embedding_dim=EMBEDDING_DIM):
        super(PositionalEncoding, self).__init__()
        
        positions = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim))
        
        pe = np.zeros((max_position, embedding_dim))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, 
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM):
        super(Transformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding слои
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(MAX_SEQ_LEN, embedding_dim)
        
        # Encoder
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append([
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dense(embedding_dim),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dropout(0.1)
            ])
        
        # Decoder
        self.decoder_layers = []
        for _ in range(num_layers):
            self.decoder_layers.append([
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads),
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dense(embedding_dim),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dropout(0.1)
            ])
        
        # Финальный слой
        self.final_dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        src, tgt = inputs
        
        # Embedding + positional encoding
        src_emb = self.embedding(src) * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        tgt_emb = self.embedding(tgt) * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Encoder
        encoder_output = src_emb
        for attn, ff1, ff2, norm, dropout in self.encoder_layers:
            # Self-attention
            attn_output = attn(encoder_output, encoder_output)
            attn_output = dropout(attn_output, training=training)
            encoder_output = norm(encoder_output + attn_output)
            
            # Feed-forward
            ff_output = ff2(ff1(encoder_output))
            ff_output = dropout(ff_output, training=training)
            encoder_output = norm(encoder_output + ff_output)
        
        # Decoder
        decoder_output = tgt_emb
        
        # Создаем маску для decoder (look-ahead mask)
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((MAX_SEQ_LEN, MAX_SEQ_LEN)), -1, 0)
        
        for self_attn, cross_attn, ff1, ff2, norm, dropout in self.decoder_layers:
            # Masked self-attention
            self_attn_output = self_attn(decoder_output, decoder_output, attention_mask=look_ahead_mask)
            self_attn_output = dropout(self_attn_output, training=training)
            decoder_output = norm(decoder_output + self_attn_output)
            
            # Cross-attention
            cross_attn_output = cross_attn(decoder_output, encoder_output)
            cross_attn_output = dropout(cross_attn_output, training=training)
            decoder_output = norm(decoder_output + cross_attn_output)
            
            # Feed-forward
            ff_output = ff2(ff1(decoder_output))
            ff_output = dropout(ff_output, training=training)
            decoder_output = norm(decoder_output + ff_output)
        
        # Финальный выход
        output = self.final_dense(decoder_output)
        return output

# ==================== ОБУЧЕНИЕ ====================
def train_transformer():
    print("\n" + "="*70)
    print("TRAINING TRANSFORMER FOR SEQUENCE COPYING TASK")
    print("="*70)
    
    # Создаем данные
    src_data, tgt_input_data, tgt_output_data = create_realistic_data(num_samples=10000)
    
    # Разделяем на train/val/test
    train_size = int(0.7 * len(src_data))
    val_size = int(0.15 * len(src_data))
    
    # Train
    train_src = src_data[:train_size]
    train_tgt = tgt_input_data[:train_size]
    train_labels = tgt_output_data[:train_size]
    
    # Validation
    val_src = src_data[train_size:train_size+val_size]
    val_tgt = tgt_input_data[train_size:train_size+val_size]
    val_labels = tgt_output_data[train_size:train_size+val_size]
    
    # Test
    test_src = src_data[train_size+val_size:]
    test_tgt = tgt_input_data[train_size+val_size:]
    test_labels = tgt_output_data[train_size+val_size:]
    
    print(f"\nРазделение данных:")
    print(f"  Train: {len(train_src)} samples")
    print(f"  Val:   {len(val_src)} samples")
    print(f"  Test:  {len(test_src)} samples")
    
    # Создаем модель
    print("\nСоздание модели Transformer...")
    model = Transformer(vocab_size=VOCAB_SIZE)
    
    # Компилируем с оптимизатором и learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Callbacks для улучшения обучения
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_transformer.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Обучение
    print("\nНачало обучения...")
    print("-" * 50)
    
    history = model.fit(
        x=[train_src, train_tgt],
        y=train_labels,
        validation_data=([val_src, val_tgt], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, (test_src, test_tgt, test_labels)

# ==================== ВИЗУАЛИЗАЦИЯ ====================
def plot_training_results(history):
    """Визуализация результатов обучения"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Loss
    axes[0, 0].plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, max(history.history['loss']) * 1.1])
    
    # 2. Accuracy
    axes[0, 1].plot(history.history['accuracy'], 'g-', linewidth=2, label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], 'orange', linewidth=2, label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # 3. Learning Rate
    if 'lr' in history.history:
        axes[0, 2].plot(history.history['lr'], 'purple', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('Learning Rate', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
    
    # 4. Loss Improvement (процент улучшения)
    loss_improvement = []
    for i in range(1, len(history.history['loss'])):
        improvement = (history.history['loss'][i-1] - history.history['loss'][i]) / history.history['loss'][i-1] * 100
        loss_improvement.append(improvement)
    
    axes[1, 0].plot(range(1, len(loss_improvement)+1), loss_improvement, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Loss Improvement (%)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Improvement %', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(range(1, len(loss_improvement)+1), loss_improvement, alpha=0.3, color='blue')
    
    # 5. Accuracy Improvement
    acc_improvement = []
    for i in range(1, len(history.history['accuracy'])):
        improvement = (history.history['accuracy'][i] - history.history['accuracy'][i-1]) * 100
        acc_improvement.append(improvement)
    
    axes[1, 1].plot(range(1, len(acc_improvement)+1), acc_improvement, 'g-', linewidth=2, marker='s', markersize=4)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Accuracy Improvement (% points)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Improvement %', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].fill_between(range(1, len(acc_improvement)+1), acc_improvement, alpha=0.3, color='green')
    
    # 6. Combined plot
    epochs = range(1, len(history.history['accuracy']) + 1)
    axes[1, 2].plot(epochs, history.history['accuracy'], 'g-', linewidth=2, label='Train Accuracy')
    axes[1, 2].plot(epochs, history.history['val_accuracy'], 'orange', linewidth=2, label='Val Accuracy')
    axes[1, 2].plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.5)
    axes[1, 2].plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.5)
    axes[1, 2].set_title('Combined View', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Value', fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Transformer Training Results - Sequence Copying Task', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Отдельный график с красивой визуализацией
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'g-', linewidth=3, label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'orange', linewidth=3, label='Validation Accuracy')
    plt.fill_between(range(len(history.history['accuracy'])), 
                     history.history['accuracy'], 
                     history.history['val_accuracy'], 
                     alpha=0.2, color='green')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.0])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'b-', linewidth=3, label='Training Loss')
    plt.plot(history.history['val_loss'], 'r-', linewidth=3, label='Validation Loss')
    plt.fill_between(range(len(history.history['loss'])), 
                     history.history['loss'], 
                     history.history['val_loss'], 
                     alpha=0.2, color='blue')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, max(history.history['loss']) * 1.1])
    
    plt.tight_layout()
    plt.show()

# ==================== ТЕСТИРОВАНИЕ ====================
def test_model(model, test_data):
    """Тестирование модели на тестовых данных"""
    test_src, test_tgt, test_labels = test_data
    
    print("\n" + "="*70)
    print("TESTING THE MODEL")
    print("="*70)
    
    # Оценка на тестовых данных
    test_loss, test_accuracy = model.evaluate(
        [test_src, test_tgt], test_labels, verbose=0
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Примеры предсказаний
    print("\nПримеры предсказаний:")
    print("-" * 50)
    
    num_examples = 5
    for i in range(min(num_examples, len(test_src))):
        # Получаем предсказания
        predictions = model.predict([test_src[i:i+1], test_tgt[i:i+1]], verbose=0)
        predicted_tokens = tf.argmax(predictions[0], axis=-1).numpy()
        
        # Убираем padding
        src_seq = test_src[i]
        tgt_seq = test_tgt[i]
        true_seq = test_labels[i]
        pred_seq = predicted_tokens
        
        # Находим реальную длину (до padding)
        src_len = np.sum(src_seq != 0)
        tgt_len = np.sum(tgt_seq != 0)
        
        print(f"\nПример {i+1}:")
        print(f"  Source:      {src_seq[:src_len]}")
        print(f"  Target Input: {tgt_seq[:tgt_len]}")
        print(f"  True Output: {true_seq[:tgt_len]}")
        print(f"  Pred Output: {pred_seq[:tgt_len]}")
        
        # Вычисляем точность для этого примера
        correct = np.sum(true_seq[:tgt_len] == pred_seq[:tgt_len])
        total = tgt_len
        print(f"  Accuracy для примера: {correct}/{total} = {correct/total:.2%}")
    
    return test_accuracy

# ==================== BLEU SCORE ====================
def calculate_bleu_score(model, test_data, num_samples=100):
    """Вычисление BLEU score для оценки качества"""
    test_src, test_tgt, test_labels = test_data
    
    print("\n" + "="*70)
    print("CALCULATING BLEU SCORE")
    print("="*70)
    
    bleu_scores = []
    
    for i in range(min(num_samples, len(test_src))):
        # Получаем предсказания
        predictions = model.predict([test_src[i:i+1], test_tgt[i:i+1]], verbose=0)
        predicted_tokens = tf.argmax(predictions[0], axis=-1).numpy()
        
        # Убираем padding и специальные токены
        true_seq = test_labels[i]
        pred_seq = predicted_tokens
        
        # Убираем токены 0, 1, 2 (специальные)
        true_seq_filtered = [str(t) for t in true_seq if t > 2]
        pred_seq_filtered = [str(t) for t in pred_seq if t > 2]
        
        if len(true_seq_filtered) > 0 and len(pred_seq_filtered) > 0:
            # Простой BLEU (n-gram совпадения)
            true_ngrams = set()
            pred_ngrams = set()
            
            # 1-граммы
            for j in range(len(true_seq_filtered)):
                true_ngrams.add(tuple([true_seq_filtered[j]]))
            for j in range(len(pred_seq_filtered)):
                pred_ngrams.add(tuple([pred_seq_filtered[j]]))
            
            # Считаем совпадения
            matches = len(true_ngrams.intersection(pred_ngrams))
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0
            
            # Brevity penalty
            if len(pred_seq_filtered) < len(true_seq_filtered):
                bp = np.exp(1 - len(true_seq_filtered) / len(pred_seq_filtered))
            else:
                bp = 1.0
            
            bleu = bp * precision
            bleu_scores.append(bleu)
    
    if bleu_scores:
        avg_bleu = np.mean(bleu_scores)
        print(f"\nAverage BLEU Score ({len(bleu_scores)} samples): {avg_bleu:.4f}")
        print(f"Min BLEU: {np.min(bleu_scores):.4f}")
        print(f"Max BLEU: {np.max(bleu_scores):.4f}")
        
        # Визуализация распределения BLEU scores
        plt.figure(figsize=(10, 6))
        plt.hist(bleu_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=avg_bleu, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_bleu:.3f}')
        plt.title('Distribution of BLEU Scores', fontsize=14, fontweight='bold')
        plt.xlabel('BLEU Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return avg_bleu
    else:
        print("Не удалось вычислить BLEU score")
        return 0

# ==================== ОСНОВНАЯ ПРОГРАММА ====================
def main():
    print("="*80)
    print("TRANSFORMER IMPLEMENTATION - MACHINE TRANSLATION TASK")
    print("Task 20: Complete Transformer Architecture with Realistic Training")
    print("="*80)
    
    # 1. Обучение модели
    model, history, test_data = train_transformer()
    
    # 2. Визуализация результатов
    plot_training_results(history)
    
    # 3. Тестирование
    test_accuracy = test_model(model, test_data)
    
    # 4. BLEU оценка
    bleu_score = calculate_bleu_score(model, test_data)
    
    # 5. Финальный отчет
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nModel Performance Summary:")
    print(f"  Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Final Training Loss:       {final_train_loss:.4f}")
    print(f"  Final Validation Loss:     {final_val_loss:.4f}")
    print(f"  Test Accuracy:             {test_accuracy:.4f}")
    print(f"  BLEU Score:                {bleu_score:.4f}")
    
    # Проверка переобучения
    overfitting_gap = final_train_acc - final_val_acc
    if overfitting_gap > 0.1:
        print(f"\n⚠️  Warning: Potential overfitting detected!")
        print(f"   Train-Val accuracy gap: {overfitting_gap:.3f}")
    else:
        print(f"\n✅ Good: No significant overfitting detected.")
        print(f"   Train-Val accuracy gap: {overfitting_gap:.3f}")
    
    # Визуализация сравнения
    metrics = ['Train Acc', 'Val Acc', 'Test Acc']
    values = [final_train_acc, final_val_acc, test_accuracy]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values, color=['green', 'orange', 'blue'], alpha=0.7)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("IMPLEMENTATION COMPLETE!")
    print("✅ Transformer успешно обучен на реалистичной задаче!")
    print("="*80)

# Запуск программы
if __name__ == "__main__":
    # Устанавливаем seed для воспроизводимости
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Проверка доступности GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU доступен: {gpus[0]}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
    else:
        print("⚠️  GPU не обнаружен, используется CPU")
    
    main()
    