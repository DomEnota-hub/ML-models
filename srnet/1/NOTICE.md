## SRNet specular highlight removal (bundled offline model)

Original source and pretrained weights: https://github.com/zhiyiYo/Specular-Removal
Original copyright: Copyright (c) 2021 Huang Zhengzhi.
Original code and pretrained checkpoint distributed in the project's MIT-licensed
repository (LICENSE): https://github.com/zhiyiYo/Specular-Removal/blob/6f44a130d2b20568f1a1b1159e32a4ec6b46ef19/LICENSE
Upstream pinned commit: `6f44a130d2b20568f1a1b1159e32a4ec6b46ef19`.
Original architecture: `specular_removal/network.py`.
Original architecture SHA-256: `472103bbc1bc6e3ab6839af147e12d3edec1110d3a8da891929c7c571a5b783f`.
Original checkpoint: `model/SRNet_2021-07-23_23-00-22.pth` (259,399 bytes).
Original checkpoint SHA-256: `d20a8b1021cd150cba0fce6738fac997f9283390de134756ecc9710245c2cba8`.
Our verified ONNX conversion: `app/src/main/assets/ai/specular_srnet_256.onnx.b64`
(base64 source of binary with SHA-256 `1538f5800858c329e93576566d40f568fcc07d8fdc674bca454fbfc4771c63e4`,
230,221 bytes). Conversion CI: https://github.com/DomEnota-hub/ML/actions/runs/42581463302
The model and source were published under the MIT License:

Copyright (c) 2021 Huang Zhengzhi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
User explicitly installs the bundled model into private app storage; no photo
leaves the Android device. This model targets specular highlights, not objects,
glassy reflections or irrevocably clipped overexposure. Quality is not guaranteed.
