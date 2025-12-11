struct Shadow {
    proj_view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> shadow: Shadow;

struct Chunk {
    position: vec3<f32>,
};
@group(1) @binding(0)
var<uniform> chunk: Chunk;

struct VertexInput {
    /// voxel_id | normal_index | z | y | x
    /// ---------+--------------+---+---+---
    /// 11       | 3            | 6 | 6 | 6
    @location(0) packed: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
	let x = f32(in.packed & 0x3f);
	let y = f32((in.packed >> 6) & 0x3f);
	let z = f32((in.packed >> 12) & 0x3f);

    var out: VertexOutput;
	let model_position = vec3(x, y, z) + chunk.position;
    out.clip_position = shadow.proj_view * vec4(model_position, 1.0);
    return out;
}
