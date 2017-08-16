import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.word_vector_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')
        xx_seq_len = tf.ones_like(x_len)*JX
        qq_seq_len = tf.ones_like(q_len)*JQ

        hidden_dim = 200
        x_fw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)
        x_bw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)
        q_fw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)
        q_bw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)

        xx_rnn, states = tf.nn.bidirectional_dynamic_rnn(x_fw, x_bw, xx, sequence_length=xx_seq_len, dtype=tf.float32)
        x_rnn = tf.concat(xx_rnn, 2)
        qq_rnn, states = tf.nn.bidirectional_dynamic_rnn(q_fw, q_bw, qq, sequence_length=qq_seq_len, dtype=tf.float32)
        q_rnn = tf.concat(qq_rnn, 2)
        
        qq_avg = tf.reduce_mean(q_rnn, axis=1)
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])
        print("[batch_size, JX, 2 * hidden_size] : ", qq_avg_exp.shape, qq_avg_tiled.shape)

        xq = tf.concat([x_rnn, qq_avg_tiled, x_rnn * qq_avg_tiled], axis=2)
        xq_flat = tf.reshape(xq, [-1, 6 * hidden_dim])
        print(xq.shape, xq_flat.shape)

        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)
            yp1 = tf.argmax(logits1, axis=1)
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)
            yp2 = tf.argmax(logits2, axis=1)

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.word_vector_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')
        xx_seq_len = tf.ones_like(x_len)*JX
        qq_seq_len = tf.ones_like(q_len)*JQ

        hidden_dim = 200
        x_fw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)
        x_bw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)
        q_fw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)
        q_bw = DropoutWrapper(GRUCell(hidden_dim), input_keep_prob=0.9, output_keep_prob=1.0)

        xx_rnn, states = tf.nn.bidirectional_dynamic_rnn(x_fw, x_bw, xx, sequence_length=xx_seq_len, dtype=tf.float32)
        x_rnn = tf.concat(xx_rnn, 2)
        qq_rnn, states = tf.nn.bidirectional_dynamic_rnn(q_fw, q_bw, qq, sequence_length=qq_seq_len, dtype=tf.float32)
        q_rnn = tf.concat(qq_rnn, 2)

        q_rnn_exp = tf.expand_dims(q_rnn, axis = 1)
        q_rnn_tiled = tf.tile(q_rnn_exp, [1, JX, 1, 1])
        x_rnn_exp = tf.expand_dims(x_rnn, axis = 2)
        x_rnn_tiled = tf.tile(x_rnn_exp, [1, 1, JQ, 1])
        x_q = tf.concat([q_rnn_tiled, x_rnn_tiled, q_rnn_tiled * x_rnn_tiled], axis = 3)
        x_q_flat = tf.reshape(x_q, [-1, 6 * hidden_dim])

        x_q_softmax = tf.nn.softmax(tf.reshape(tf.layers.dense(inputs=x_q_flat, units=1), [-1, JX, JQ]), dim=-1)
        q_rnn_avg_tiled = tf.reduce_sum(q_rnn_tiled * tf.tile(tf.expand_dims(x_q_softmax, axis=3), [1, 1, 1, 2 * hidden_dim]), 2)

        xq = tf.concat([x_rnn, q_rnn_avg_tiled, x_rnn * q_rnn_avg_tiled], axis=2)
        xq_flat = tf.reshape(xq, [-1, 6 * hidden_dim])
        print(xq.shape, xq_flat.shape)

        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)
            yp1 = tf.argmax(logits1, axis=1)
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)
            yp2 = tf.argmax(logits2, axis=1)

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
