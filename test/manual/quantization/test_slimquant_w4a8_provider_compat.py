"""CPU-only provider checkpoint compatibility checks; never load a full model."""
import ast
import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import sglang.srt.layers.quantization  # Initialize the registry before staging overrides.

ROOT = Path(__file__).parent
if (ROOT / 'slimquant_w4a8.py').exists():
    for name in ('slimquant_w4a8',):
        module_name = 'sglang.srt.layers.quantization.' + name
        spec = importlib.util.spec_from_file_location(module_name, ROOT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8Config as Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod, W8A8Int8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.utils import WeightsMapper

MODEL = Path('/home/work/GLM-5.3-Flash-Int4')
RAW = json.loads((MODEL / 'config.json').read_text())['quantization_config']


class CompatibilityTests(unittest.TestCase):
    def test_int8_and_ignored_linear(self):
        q = Config.from_config(RAW)
        layer = LinearBase.__new__(LinearBase)
        for prefix in ('model.layers.44.self_attn.q_proj', 'model.layers.44.mlp.shared_experts.down_proj', 'model.layers.0.mlp.down_proj'):
            self.assertIsInstance(q.get_quant_method(layer, prefix), W8A8Int8LinearMethod)
        for prefix in ('model.layers.3.self_attn.kv_b_proj', 'model.layers.44.mlp.gate', 'visual.merger.down_proj'):
            self.assertIsInstance(q.get_quant_method(layer, prefix), UnquantizedLinearMethod)
        self.assertIsInstance(Config.from_config({}).get_quant_method(layer, 'model.layers.0.self_attn.q_proj'), UnquantizedLinearMethod)

    def test_fused_and_nextn_ignore_mapping(self):
        q = Config.from_config(RAW)
        q.apply_weight_name_mapper(WeightsMapper(orig_to_new_substr={'model.layers.45': 'model.decoder'}))
        q.update_packed_modules_mapping({'gate_up_proj': ['gate_proj', 'up_proj'], 'fused_qkv_a_proj_with_mqa': ['q_a_proj', 'kv_a_proj_with_mqa']})
        layer = LinearBase.__new__(LinearBase)
        for prefix in ('model.decoder.mlp.shared_experts.gate_up_proj', 'model.decoder.self_attn.fused_qkv_a_proj_with_mqa'):
            self.assertIsInstance(q.get_quant_method(layer, prefix), W8A8Int8LinearMethod)
        self.assertIsInstance(q.get_quant_method(layer, 'model.decoder.self_attn.kv_b_proj'), UnquantizedLinearMethod)
        self.assertIsInstance(q.get_quant_method(layer, 'visual.blocks.0.attn.qkv_proj'), UnquantizedLinearMethod)
        self.assertIsInstance(q.get_int8_config(), W8A8Int8Config)

    def test_int8_storage_and_scale_load(self):
        method = Config.from_config(RAW).get_quant_method(LinearBase.__new__(LinearBase), 'model.layers.0.self_attn.q_proj')
        layer = torch.nn.Module()
        method.create_weights(layer, 4, [3], 4, 3, torch.bfloat16, weight_loader=lambda *a: None)
        weight = torch.tensor([[-128, -1, 0, 127]] * 3, dtype=torch.int8)
        scale = torch.tensor([[0.1], [0.2], [0.3]])
        layer.weight.data.copy_(weight)
        layer.weight_scale.data.copy_(scale)
        self.assertEqual(layer.weight.dtype, torch.int8)
        torch.testing.assert_close(layer.weight.float() * layer.weight_scale, weight.float() * scale)

    def test_nextn_int8_and_bf16_resolution(self):
        source = ROOT / 'glm5_next_nextn.py'
        if source.exists():
            spec = importlib.util.spec_from_file_location('compat_nextn',source)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = mod.Glm5NextForConditionalGenerationNextN
        else:
            from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN as cls
        model = cls.__new__(cls)
        config = SimpleNamespace(num_hidden_layers=45,quantization_config=RAW)
        q = Config.from_config(RAW)
        self.assertIsInstance(model._resolve_nextn_quant_config(config,q),W8A8Int8Config)
        config.quantization_config = {'ignore':['model.layers.45.*']}
        self.assertIsNone(model._resolve_nextn_quant_config(config,Config()))

    def test_provider_packed_bytes_and_scale(self):
        name = 'model.layers.3.mlp.experts.0.gate_proj'
        packed = torch.arange(256, dtype=torch.uint8).reshape(16,16)
        mapped, result = Config.normalize_expert_weight(name + '.qweight', packed)
        self.assertEqual(mapped, name + '.weight')
        self.assertEqual(result.dtype, torch.int8)
        expected = ((packed & 15) << 4) | (packed >> 4)
        torch.testing.assert_close(result.view(torch.uint8), expected)
        scale = torch.ones((16,1), dtype=torch.float16)
        mapped, result = Config.normalize_expert_weight(name + '.scales', scale)
        self.assertEqual(mapped, name + '.weight_scale')
        self.assertIs(result, scale)

    def test_compressed_tensors_signed_values(self):
        values = torch.arange(-8,8,dtype=torch.int64).reshape(2,8)
        encoded = torch.sum((values+8) << (torch.arange(8)*4), dim=1).to(torch.int32).reshape(2,1)
        name, packed = Config.normalize_expert_weight('model.layers.3.mlp.experts.0.gate_proj.weight_packed', encoded)
        b = packed.to(torch.int32) & 255
        nibbles = torch.stack((b >> 4, b & 15), dim=-1).flatten(-2)
        decoded = torch.where(nibbles >= 8, nibbles-16, nibbles)
        torch.testing.assert_close(decoded.long(), values)
        self.assertIsNone(Config.normalize_expert_weight('model.layers.3.mlp.experts.0.gate_proj.weight_shape', torch.tensor([2,8]))[0])

    def test_swiglu_clamp(self):
        from sglang.srt.layers.quantization.slimquant_w4a8 import _apply_slimquant_activation
        gu = torch.tensor([[-20., -3., 4., 20., -20., 20., -3., 4.]])
        gate, up = gu.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10,10)
        torch.testing.assert_close(_apply_slimquant_activation(gu,'silu',10),expected)
        torch.testing.assert_close(_apply_slimquant_activation(gu,'silu'),torch.nn.functional.silu(gate)*up)

    def test_checkpoint_nibble_order(self):
        # Independent on-disk decoder: provider is signed, low-nibble-first;
        # compressed-tensors is low-nibble-first with an offset of eight.
        from safetensors import safe_open
        roots = [(MODEL,'qweight',0), (Path('/home/work/GLM-5.3-Flash-Channel-INT4-w4a16'),'weight_packed',8)]
        for root,suffix,offset in roots:
            index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
            for proj in ('gate_proj','up_proj','down_proj'):
                key = next(k for k in index if k.endswith(f'layers.3.mlp.experts.0.{proj}.{suffix}'))
                with safe_open(str(root/index[key]),framework='pt',device='cpu') as f:
                    raw = f.get_slice(key)[:16]
                b = raw.contiguous().view(torch.uint8).to(torch.int32)
                ref = torch.stack((b & 15,b >> 4),dim=-1).flatten(-2)
                ref = ref-offset if offset else torch.where(ref>=8,ref-16,ref)
                _, packed = Config.normalize_expert_weight(key,raw)
                b = packed.to(torch.int32)&255
                actual = torch.stack((b>>4,b&15),dim=-1).flatten(-2)
                actual = torch.where(actual>=8,actual-16,actual)
                torch.testing.assert_close(actual,ref)

    def test_glm_loader_routes_provider_experts(self):
        # Execute the actual load_weights body on a small parameter dictionary.
        source = ROOT / 'glm5_next.py'
        if not source.exists():
            source = Path('/home/work/code/sglang-das/python/sglang/srt/models/glm5_next.py')
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'ModelNextForCausalLM')
        fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'load_weights')
        fn.returns = None
        for arg in fn.args.args: arg.annotation = None
        calls = []
        class ExpertMapping:
            @staticmethod
            def make_expert_params_mapping(**kwargs):
                return [('experts.w13_', 'experts.0.gate_proj.', 0, 'w1')]
        namespace = dict(torch=torch, ModelNextForCausalLM=SimpleNamespace(_AWQ_LIKE_QUANT_METHOD=[], _STACKED_PARAMS_MAPPING=[]), FusedMoE=ExpertMapping)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        dummy = SimpleNamespace(config=SimpleNamespace(n_routed_experts=1,mla=False),num_fused_shared_experts=0,quant_config=Config.from_config(RAW))
        param = SimpleNamespace(weight_loader=lambda *a,**k: calls.append((a,k)))
        packed = torch.tensor([[0x8f,0x17]],dtype=torch.uint8)
        scale = torch.ones((1,1),dtype=torch.float16)
        params = {'model.layers.3.mlp.experts.w13_weight':param,'model.layers.3.mlp.experts.w13_weight_scale':param}
        namespace['load_weights'](dummy,[('model.layers.3.mlp.experts.0.gate_proj.qweight',packed),('model.layers.3.mlp.experts.0.gate_proj.scales',scale)],params_dict=params)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0][1],dict(shard_id='w1',expert_id=0))
        self.assertEqual(calls[0][0][1].dtype,torch.int8)


def audit_headers():
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    q = Config.from_config(RAW)
    layer = LinearBase.__new__(LinearBase)
    checked = {'I8':0, 'BF16':0}
    errors = []
    for shard in sorted(set(index.values())):
        with (MODEL/shard).open('rb') as f:
            header = json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
        for key,value in header.items():
            if key == '__metadata__' or not key.endswith('.weight') or '.experts.' in key:
                continue
            if len(value['shape']) != 2 or value['dtype'] not in checked:
                continue
            if not any(s in key for s in ('.self_attn.', '.mlp.', '.visual.')):
                continue
            name = key.removesuffix('.weight').replace('language_model.','').replace('model.visual.','visual.')
            method = q.get_quant_method(layer,name)
            expected = W8A8Int8LinearMethod if value['dtype']=='I8' else UnquantizedLinearMethod
            checked[value['dtype']]+=1
            if not isinstance(method,expected): errors.append((name,value['dtype'],type(method).__name__))
    print('HEADER_AUDIT', checked, 'errors', errors, flush=True)
    assert not errors

if __name__ == '__main__':
    program = unittest.main(exit=False)
    if not program.result.wasSuccessful():
        raise SystemExit(1)
    audit_headers()
