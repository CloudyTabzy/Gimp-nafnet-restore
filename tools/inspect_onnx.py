import onnx
m = onnx.load(r'C:\Dev\GIMP_Native_Plugin\Gimp-restoration-plugin\NAFNet-REDS-width64_v1.onnx')
print('Producer:', m.producer_name, m.producer_version)
print('Opset:', m.opset_import[0].version)
print('IR version:', m.ir_version)
print('Inputs:')
for inp in m.graph.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(' ', inp.name, dims, 'dtype=', onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type))
print('Outputs:')
for o in m.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
    print(' ', o.name, dims, 'dtype=', onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type))
print('Nodes:', len(m.graph.node))
print('Producer metadata props:')
for p in m.producer_props:
    print(' ', p.key, '=', p.value if len(p.value) < 100 else p.value[:100] + '...')
