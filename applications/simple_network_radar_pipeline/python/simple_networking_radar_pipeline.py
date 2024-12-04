# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time

try:
    import numpy as np
except ImportError:
    raise ImportError("This demo requires numpy, but it could not be imported.")

try:
    import cupy as cp
except ImportError:
    raise ImportError("This demo requires cupy, but it could not be imported.")

try:
    import cusignal
except ImportError:
    raise ImportError("This demo requires cusignal, but it could not be imported.")

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.decorator import Input, Output, create_op

import matplotlib.pyplot as plt

import multiprocessing as mp

queue = mp.Queue()

def visualize(queue):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, 1000)
    ax.set_ylim(-3, 3)

    while True:
        if not queue.empty():
            data = queue.get()
            line.set_data(range(len(data)), data)
            fig.canvas.draw()
            fig.canvas.flush_events()


# Radar Settings
num_channels = 16
num_pulses = 128
num_uncompressed_range_bins = 9000
waveform_length = 1000
num_compressed_range_bins = num_uncompressed_range_bins - waveform_length + 1
NDfft = 256
Pfa = 1e-5
iterations = 100


# Window Settings
window = cusignal.hamming(waveform_length)
# The -2 is a hack here to account for a 3 tap MTI filter
range_doppler_window = cp.transpose(
    cp.repeat(
        cp.expand_dims(cusignal.hamming(num_pulses - 2), 0), num_compressed_range_bins, axis=0
    )
)
Nfft = 2 ** math.ceil(math.log2(np.max([num_uncompressed_range_bins, waveform_length])))

# CFAR Settings
mask = cp.transpose(
    cp.asarray(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
)

# unsure how to use the "count" condition on this operator
@create_op(
    outputs=("x","waveform"))
def signal_generator(count=1000):
    for _ in range(count):
        x = cp.random.randn(
            num_pulses, num_uncompressed_range_bins, dtype=cp.float32
        ) + 1j * cp.random.randn(num_pulses, num_uncompressed_range_bins, dtype=cp.float32)
        waveform = cp.random.randn(waveform_length, dtype=cp.float32) + 1j * cp.random.randn(
            waveform_length, dtype=cp.float32
        )
        yield (x, waveform)

@create_op(inputs=("x","waveform"))
def viz_waveform(x, waveform):
    queue.put(np.real(waveform.get()))
    # print(f"viz data y({waveform.shape})=", waveform[:3], "...")

# class SignalGeneratorOp(Operator):
#     def __init__(self, *args, **kwargs):
#         # Need to call the base class constructor last
#         super().__init__(*args, **kwargs)

#     def setup(self, spec: OperatorSpec):
#         spec.output("x")
#         spec.output("waveform")

#     def compute(self, op_input, op_output, context):
#         x = cp.random.randn(
#             num_pulses, num_uncompressed_range_bins, dtype=cp.float32
#         ) + 1j * cp.random.randn(num_pulses, num_uncompressed_range_bins, dtype=cp.float32)
#         waveform = cp.random.randn(waveform_length, dtype=cp.float32) + 1j * cp.random.randn(
#             waveform_length, dtype=cp.float32
#         )

#         op_output.emit(x, "x")
#         op_output.emit(waveform, "waveform")

@create_op(
    inputs=("x", "waveform"),
    outputs=("X")
)
def pulse_compression(x, waveform):
    waveform_windowed = waveform * window
    waveform_windowed_norm = waveform_windowed / cp.linalg.norm(waveform_windowed)

    W = cp.conj(cp.fft.fft(waveform_windowed_norm, Nfft))
    X = cp.fft.fft(x, Nfft, 1)

    for pulse in range(num_pulses):
        y = cp.fft.ifft(cp.multiply(X[pulse, :], W), Nfft, 0)
        x[pulse, 0:num_compressed_range_bins] = y[0:num_compressed_range_bins]

    x_compressed = x[:, 0:num_compressed_range_bins]

    x_compressed_stack = cp.stack([x_compressed] * num_channels)
    return x_compressed_stack

class PulseCompressionOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        self.index = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("x")
        spec.input("waveform")
        spec.output("X")

    def compute(self, op_input, op_output, context):
        x = op_input.receive("x")
        waveform = op_input.receive("waveform")

        waveform_windowed = waveform * window
        waveform_windowed_norm = waveform_windowed / cp.linalg.norm(waveform_windowed)

        W = cp.conj(cp.fft.fft(waveform_windowed_norm, Nfft))
        X = cp.fft.fft(x, Nfft, 1)

        for pulse in range(num_pulses):
            y = cp.fft.ifft(cp.multiply(X[pulse, :], W), Nfft, 0)
            x[pulse, 0:num_compressed_range_bins] = y[0:num_compressed_range_bins]

        x_compressed = x[:, 0:num_compressed_range_bins]

        x_compressed_stack = cp.stack([x_compressed] * num_channels)

        op_output.emit(x_compressed_stack, "X")

@create_op(
    inputs=("x"),
    outputs=("X"))
def mti_filter(x):
    for channel in range(num_channels):
        x_conv2 = cusignal.convolve2d(x[channel, :, :], cp.array([[1], [-2], [-1]]), "valid")
        x_conv2_stack = cp.stack([x_conv2] * num_channels)

    return x_conv2_stack

@create_op(
    inputs=("x"),
    outputs=("X"))
def range_doppler(x):
    for channel in range(num_channels):
        x_window = cp.fft.fft(cp.multiply(x[channel, :, :], range_doppler_window), NDfft, 0)
        x_window_stack = cp.stack([x_window] * num_channels)

    return x_window_stack

@create_op(
    inputs=("x"),
    outputs=("X"))
def cfar_fn(x):
    norm = cusignal.convolve2d(cp.ones(x.shape[1::]), mask, "same")

    for channel in range(num_channels):
        background_averages = cp.divide(
            cusignal.convolve2d(cp.abs(x[channel, :, :]) ** 2, mask, "same"), norm
        )
        alpha = cp.multiply(norm, cp.power(Pfa, cp.divide(-1.0, norm)) - 1)
        dets = cp.zeros(x[channel, :, :].shape)
        dets[cp.where(cp.abs(x[channel, :, :]) ** 2 > cp.multiply(alpha, background_averages))]

        dets_stacked = cp.stack([dets] * num_channels)

    return dets_stacked

class SinkOp(Operator):
    def __init__(self, *args, shape=(512, 512), **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("X")

    def compute(self, op_input, op_output, context):
        op_input.receive("X")


class BasicRadarFlow(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        # src = SignalGeneratorOp(self, CountCondition(self, iterations), name="src")
        src = signal_generator(self, count=iterations, name="src")
        pulseCompression = pulse_compression(self, name="pulse-compression")
        mtiFilter = mti_filter(self, name="mti-filter")
        rangeDoppler = range_doppler(self, name="range-doppler")
        cfar = cfar_fn(self, name="cfar")

        sink = SinkOp(self, name="sink")

        viz = viz_waveform(self, name="viz")

        self.add_flow(src, pulseCompression, {("x", "x"), ("waveform", "waveform")})
        self.add_flow(pulseCompression, mtiFilter)
        self.add_flow(mtiFilter, rangeDoppler)
        self.add_flow(rangeDoppler, cfar)
        self.add_flow(cfar, sink)

        self.add_flow(src, viz, {("x", "x"), ("waveform", "waveform")})

if __name__ == "__main__":
    app = BasicRadarFlow()
    app.config("")

    viz_process = mp.Process(target=visualize, args=(queue,))
    viz_process.start()

    tstart = time.time()
    app.run()
    tstop = time.time()

    duration = (iterations * num_pulses * num_channels) / (tstop - tstart)

    print(f"{duration:0.3f} pulses/sec")

    viz_process.kill()
