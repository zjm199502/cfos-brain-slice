"""奥林巴斯 / Evident cellSens .vsi 元数据解析（.vsi 文件里的树形结构：图像尺寸、维度顺序、通道名、发射波长、像素尺寸、z 位置）。

本目录的代码取自 bioio-vsi（https://github.com/szarma/bioio-vsi ，提交 2dd575b，2026-10-16，BSD-3-Clause，
版权和许可见 LICENSE.txt），只做了两处改动：导入路径改为相对导入；新增 parse_all_images（文件里有多幅图时逐幅解析，
单幅解析失败不影响其他图），并在同一 stack 有多个 .ets 时取第一个。像素读取（.ets 图块）在 cfos/core/vsi.py 里另写。"""
from ._tree import load_tree  # noqa: F401
from ._vsi import ImageVolumeMeta, parse_all_images  # noqa: F401
from .errors import VsiError  # noqa: F401

# ---- bioio-vsi 的许可（BSD-3-Clause），与 LICENSE.txt 相同，随代码一起分发 ----
# BSD 3-Clause License
#
# Copyright (c) 2026, Srdjan Sarikas
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
