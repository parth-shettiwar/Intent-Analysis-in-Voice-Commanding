from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense
from tensorflow.keras.layers import InputSpec

from extras import ReusableEmbedding, TiedOutputEmbedding
from position import TransformerCoordinateEmbedding
from transformer import TransformerACT, TransformerBlock

def vanilla_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):
    """
    A model which is almost identical to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it uses L2 regularization of the word embedding matrix,
    instead of the dropout.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        1,
        name='coordinate_embedding')
    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)

    next_step_input = coordinate_embedding_layer(next_step_input, step=0)
    for i in range(transformer_depth):
        next_step_input = (
            TransformerBlock(
                name='transformer' + str(i), num_heads=num_heads,
                residual_dropout=transformer_dropout,
                attention_dropout=transformer_dropout,
                use_masking=True,
                vanilla_wiring=True)
            (next_step_input))

    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model