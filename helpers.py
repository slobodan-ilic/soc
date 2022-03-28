import os
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import tensorflow as tf
import tensorflow.python.keras as keras
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as K

import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    BBoxSplitter,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from sentinelhub import geo_utils as gu

shc = SHConfig()
shc.sh_client_id = "24550f2c-330b-4dff-ab0f-d4f095b855ac"
shc.sh_client_secret = "|g>;cD:N3CfP<hkq-CPh6n2*xC<}57*xlr2!C?e~"
shc.save()


EVAL_SCRIPT_ALL_BANDS = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B05",
                        "B06", "B07", "B08", "B09",
                        "B10", "B11", "B12", "B8A"
                ],
                units: "DN",
            }],
            output: {
                bands: 13,
                sampleType: "INT16"
            }
        };
    }
    function evaluatePixel(sample) {
        return [
            sample.B01, sample.B02, sample.B03, sample.B04, sample.B05,
            sample.B06, sample.B07, sample.B08, sample.B09,
            sample.B10, sample.B11, sample.B12, sample.B8A
        ];
    }
"""


class SentinelHelper:
    """Implement relevant functions for working with Sentinel Hub."""

    def __init__(
        self, wgs84_bbox, quartal, year, n_rows_div=3, n_cols_div=3, resolution=10
    ):
        self._wgs84_bbox = wgs84_bbox
        self._quartal = quartal
        self._year = year
        self._n_rows_div = n_rows_div
        self._n_cols_div = n_cols_div
        self._resolution = resolution

    # ---------------------------- API -------------------------------------------------

    def image(self) -> np.ndarray:
        return self._assembled_img()

    # ---------------------------- IMPLEMENTATION --------------------------------------

    def _assembled_img(self):
        """return np.array, assembled from bbox patches."""
        patches = self._img_patches()
        ind = np.flip(
            np.arange(self._n_rows_div * self._n_cols_div)
            .reshape(self._n_rows_div, self._n_cols_div)
            .transpose(),
            axis=0,
        )
        (h, w, _) = patches[0].shape
        row_patches = []
        for i in range(self._n_rows_div):
            col_patches = []
            for j in range(self._n_cols_div):
                k = ind[i, j]
                col_patches.append(patches[k])
            row_patch = np.stack(col_patches, axis=1).reshape(
                h, w * self._n_cols_div, 13
            )
            row_patches.append(row_patch)
        return np.stack(row_patches, axis=0).reshape(
            h * self._n_rows_div, w * self._n_cols_div, 13
        )

    def _bbox_splitter(self) -> BBoxSplitter:
        """BBoxSplitter from a larger bbox, using UTM coordinates."""
        bbox = BBox(bbox=self._utm_bbox(), crs=CRS.UTM_34N)
        return BBoxSplitter(
            [bbox.geometry], CRS.UTM_34N, (self._n_rows_div, self._n_cols_div)
        )

    def _time_interval(self):
        """tuple(str, str) for sentinel request.
        It has to be in the format: time_interval=("2021-06-01", "2021-09-30")."""
        quartals = {1: ["01", "03"], 2: ["04", "06"], 3: ["07", "09"], 4: ["10", "12"]}
        [start_month, end_month] = quartals[self._quartal]
        start_date_str = f"{self._year}-{start_month}-01"
        end_date_str = f"{self._year}-{end_month}-30"
        return start_date_str, end_date_str

    def _img_patches(self):
        bblist = self._bbox_splitter().get_bbox_list()
        patches = []
        for bbox in bblist:
            request_all_bands = SentinelHubRequest(
                evalscript=EVAL_SCRIPT_ALL_BANDS,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L1C,
                        time_interval=self._time_interval(),
                        mosaicking_order="leastCC",
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, resolution=self._resolution),
                config=shc,
            )
            all_bands_response = request_all_bands.get_data()
            print(
                f"Returned data is of type = "
                f"{type(all_bands_response)} and length {len(all_bands_response)}."
            )
            print(
                f"Single element in the list is"
                f" of type {type(all_bands_response[-1])} and"
                f" has shape {all_bands_response[-1].shape}"
            )
            patches.append(all_bands_response[-1])
        return patches

    def _utm_bbox(self):
        """tuple of UTM bbox coordinates."""
        crs = CRS.UTM_34N
        print(f"BBOX: {self._wgs84_bbox}")
        min_lon, min_lat, max_lon, max_lat = self._wgs84_bbox
        min_lon_utm, min_lat_utm = gu.wgs84_to_utm(min_lon, min_lat, utm_crs=crs)
        max_lon_utm, max_lat_utm = gu.wgs84_to_utm(max_lon, max_lat, utm_crs=crs)
        return min_lon_utm, min_lat_utm, max_lon_utm, max_lat_utm


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
