"""Research-only byte-range reconstruction. Never publish checkpoint bytes or models."""
import hashlib
import mmap
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def tensors(message, path=()):
    # Traverse every populated protobuf message, including nested Constant attributes.
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield path, message
        return
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.label == field.LABEL_REPEATED:
            for i, child in enumerate(value):
                yield from tensors(child, path + (field.name, i))
        else:
            yield from tensors(value, path + (field.name,))

def plan(reference, control, sources):
    """Only raw payloads that differ from an independent untrained export may be patched.
    Every other protobuf byte must be identical. Unsupported folding fails closed."""
    import onnx
    a, b = onnx.load_model_from_string(reference), onnx.load_model_from_string(control)
    ta, tb = dict(tensors(a)), dict(tensors(b))
    if ta.keys() != tb.keys():
        raise ValueError("Export tensor structure differs")
    varying = {}
    for key, x in ta.items():
        y = tb[key]
        if x.SerializeToString() == y.SerializeToString():
            continue
        raw, other = x.raw_data, y.raw_data
        if not raw or len(raw) != len(other):
            raise ValueError("Unsupported tensor representation or size")
        x.raw_data = bytes(len(raw))
        y.raw_data = bytes(len(other))
        if x.SerializeToString() != y.SerializeToString():
            raise ValueError("Tensor metadata differs")
        varying.setdefault(raw, []).append(key)
    if not varying:
        raise ValueError("No learned differences found")
    if a.SerializeToString() != b.SerializeToString():
        raise ValueError("Non-payload graph bytes differ")
    ranges = []
    unmatched = []
    for raw, keys in varying.items():
        positions = []
        pos = 0
        while True:
            pos = reference.find(raw, pos)
            if pos < 0:
                break
            positions.append(pos)
            pos += len(raw)
            if len(positions) > len(keys):
                break
        if len(positions) != len(keys):
            unmatched.append(("ambiguous output", len(raw)))
            continue
        source = next(((name, offset) for name, content in sources.items()
                       if (offset := content.find(raw)) >= 0), None)
        if source is None:
            unmatched.append(("not contiguous in original checkpoint", len(raw)))
            continue
        for pos in positions:
            ranges.append(dict(outputOffset=pos, source=source[0], sourceOffset=source[1],
                               length=len(raw), sha256=sha(raw)))
    if unmatched:
        raise ValueError("Unmapped payloads: " + str(len(unmatched)) +
                         "; bytes=" + str(sum(x[1] for x in unmatched)) +
                         "; reasons=" + str(sorted(set(x[0] for x in unmatched))))
    ranges.sort(key=lambda x: x["outputOffset"])
    template = bytearray(reference)
    previous = 0
    for r in ranges:
        start, end = r["outputOffset"], r["outputOffset"] + r["length"]
        if start < previous:
            raise ValueError("Overlapping tensor payloads")
        template[start:end] = bytes(r["length"])
        previous = end
    # Parse the actual zeroed binary rather than trusting the substring locations.
    if onnx.load_model_from_string(bytes(template)).SerializeToString() != a.SerializeToString():
        raise ValueError("Patched bytes do not match audited graph skeleton")
    return bytes(template), ranges

def reconstruct(template, ranges, sources, expected):
    out = bytearray(template)
    previous = 0
    for r in ranges:
        start, length, offset = r["outputOffset"], r["length"], r["sourceOffset"]
        end = start + length
        source = sources[r["source"]]
        if length <= 0 or start < previous or end > len(out) or offset < 0 or offset + length > len(source):
            raise ValueError("Invalid range")
        if any(out[start:end]):
            raise ValueError("Template payload is not empty")
        data = source[offset:offset + length]
        if sha(data) != r["sha256"]:
            raise ValueError("Source slice digest mismatch")
        out[start:end] = data
        previous = end
    if sha(out) != expected:
        raise ValueError("Reconstructed model hash mismatch")
    return bytes(out)

def check_synthetic():
    import onnx
    from onnx import helper, TensorProto
    import struct
    def model(values):
        t = TensorProto(name="w", data_type=TensorProto.FLOAT, dims=[4], raw_data=struct.pack("<4f", *values))
        g = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])], "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])], [t])
        return helper.make_model(g).SerializeToString()
    real, control = model([1.125, 2.25, 3.5, 4.75]), model([7.25, 8.5, 9.75, 10.125])
    raw = struct.pack("<4f", 1.125, 2.25, 3.5, 4.75)
    sources = {"test": b"prefix" + raw + b"suffix"}
    template, ranges = plan(real, control, sources)
    assert len(ranges) == 1 and reconstruct(template, ranges, sources, sha(real)) == real
    for invalid in ({"test": b"prefix" + bytes(16) + b"suffix"}, {}):
        try:
            plan(real, control, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unmapped input accepted")
    try:
        reconstruct(template, ranges, {"test": b"prefix" + bytes(16) + b"suffix"}, sha(real))
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted slice accepted")
    print("SYNTHETIC_BYTE_RANGE_TESTS_PASS", flush=True)

def main():
    import contextlib
    import importlib.util
    import json
    import subprocess
    import sys
    import tempfile
    import types
    import urllib.request
    import gdown
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision.models import vgg19
    import onnx
    check_synthetic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    recipe = json.loads((Path(__file__).parent / "recipe-v1.json").read_text())
    with tempfile.TemporaryDirectory(prefix="dexnet-probe-") as tmp:
        root = Path(tmp)
        src = root / "source"
        subprocess.run(["git", "clone", "-q", "--no-checkout", recipe["architecture"]["repository"], str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", recipe["architecture"]["revision"]], check=True)
        if subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"]).decode().strip() != recipe["architecture"]["revision"]:
            raise RuntimeError("source mismatch")
        dex, vgg = root / "dex.pt", root / "vgg.pth"
        if not gdown.download(id="1jkhCvyrQAQZtLaxdDWDkkSOF0NZ12aJp", output=str(dex), quiet=True):
            raise RuntimeError("original DExNet unavailable")
        with urllib.request.urlopen(recipe["inputs"][1]["originalSource"], timeout=90) as response, vgg.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
                if out.tell() > 600_000_000:
                    raise ValueError("VGG19 input too large")
        for item, path in zip(recipe["inputs"], (dex, vgg)):
            if file_sha(path) != item["sha256"] or ("bytes" in item and path.stat().st_size != item["bytes"]):
                raise ValueError("original checkpoint mismatch: " + item["id"])
            print("ORIGINAL_CHECKPOINT_VERIFIED", item["id"], path.stat().st_size, flush=True)

        sys.modules["models"] = types.ModuleType("models")
        sys.modules["models"].__path__ = [str(src / "models")]
        sys.modules["models.arch"] = types.ModuleType("models.arch")
        sys.modules["models.arch"].__path__ = [str(src / "models/arch")]
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, src / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        load("models.arch.invNet", "models/arch/invNet.py")
        arch = load("models.arch.DexNet", "models/arch/DexNet.py").DExNet_expand
        class CompleteModel(nn.Module):
            def __init__(self, vgg, network):
                super().__init__()
                self.vgg, self.network = vgg, network
            def forward(self, rgb):
                features, x = [], rgb
                for index, layer in enumerate(self.vgg):
                    x = layer(x)
                    if index + 1 in (2, 7, 12, 21, 30):
                        features.append(F.interpolate(x, size=rgb.shape[2:], mode="bilinear", align_corners=False))
                transmission, reflection, _, _ = self.network(rgb, torch.cat((rgb, *features), dim=1))
                return transmission[-1], reflection[-1]
        def build(trained):
            torch.manual_seed(20261126)
            network = arch(1475, 3)
            extractor = vgg19(weights=None)
            if trained:
                state = torch.load(dex, weights_only=True, map_location="cpu")
                network.load_state_dict(state["weight"], strict=True)
                del state
                state = torch.load(vgg, weights_only=True, map_location="cpu")
                extractor.load_state_dict(state, strict=True)
                del state
            return CompleteModel(extractor.features[:30], network).eval()
        def export(trained, path):
            model = build(trained)
            with torch.no_grad():
                torch.onnx.export(model, torch.full((1, 3, 128, 128), .25), str(path),
                    opset_version=17, do_constant_folding=True, input_names=["image"],
                    output_names=["transmission", "reflection"], export_params=True)
            onnx.checker.check_model(str(path))
        real, control = root / "real.onnx", root / "control.onnx"
        export(True, real)
        actual = file_sha(real)
        print("REFERENCE_EXPORT", real.stat().st_size, actual, flush=True)
        if actual != recipe["export"]["expectedSha256"] or real.stat().st_size != recipe["export"]["expectedBytes"]:
            raise ValueError("Existing Android pin not reproduced; do not change app pin")
        export(False, control)
        with contextlib.ExitStack() as stack:
            sources = {name: stack.enter_context(mmap.mmap(stack.enter_context(path.open("rb")).fileno(), 0, access=mmap.ACCESS_READ))
                       for name, path in (("dexnet", dex), ("vgg19", vgg))}
            template, ranges = plan(real.read_bytes(), control.read_bytes(), sources)
            restored = reconstruct(template, ranges, sources, actual)
            assert restored == real.read_bytes()
            print("BYTE_EXACT_LOCAL_RECONSTRUCTION_PASS", len(ranges), sum(r["length"] for r in ranges), flush=True)
            print("No model/template/checkpoint/patch artifact published.", flush=True)

if __name__ == "__main__":
    main()
