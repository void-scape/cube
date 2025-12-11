struct Camera {
    proj_view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Light {
	view_proj: mat4x4<f32>,
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(1) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) packed: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
	let x = f32(in.packed & 0x3f);
	let y = f32((in.packed >> 6) & 0x3f);
	let z = f32((in.packed >> 12) & 0x3f);
	let color_index = (in.packed >> 21) & 1;

	const COLORS = array<vec3<f32>, 2>(
		vec3(1.0, 1.0, 1.0),
		vec3(0.0, 0.0, 0.0),
	);

    var out: VertexOutput;
    out.clip_position = camera.proj_view * vec4(light.position + vec3(x, y, z), 1.0);
	out.color = COLORS[color_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
