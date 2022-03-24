import os
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import tensorflow as tf
from tensorflow.python.keras import backend as K


class NpuHelperForTF:
    """Initialize NPU session for TF on Ascend platform."""

    def __init__(self, device_id, rank_id, rank_size, job_id, rank_table_file):
        # Init Ascend
        os.environ["ASCEND_DEVICE_ID"] = device_id
        os.environ["JOB_ID"] = job_id
        os.environ["RANK_ID"] = rank_id
        os.environ["RANK_SIZE"] = rank_size
        os.environ["RANK_TABLE_FILE"] = rank_table_file

        sess_config = tf.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        custom_op.parameter_map["graph_run_mode"].i = 0
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess_config.graph_options.rewrite_options.memory_optimization = (
            RewriterConfig.OFF
        )
        self._sess = tf.Session(config=sess_config)
        K.set_session(self._sess)

    def sess(self):
        return self._sess
