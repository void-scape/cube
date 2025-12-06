struct Camera {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(1) @binding(0)
var<uniform> light: Light;

struct Chunk {
    position: vec3<f32>,
};
@group(2) @binding(0)
var<uniform> chunk: Chunk;

struct VertexInput {
    /// color_index | normal_index | z | y | x
    /// ---------------------------------------
    /// 1             3              5   8   5
    @location(0) packed: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
	let x = f32(in.packed & 0x1f);
	let y = f32((in.packed >> 5) & 0xff);
	let z = f32((in.packed >> 13) & 0x1f);
	let normal_index = (in.packed >> 18) & 3;
	let color_index = (in.packed >> 21) & 1;

	const NORMALS = array<vec3<f32>, 6>(
		vec3(1.0, 0.0, 0.0),
		vec3(-1.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, -1.0, 0.0),
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, -1.0),
	);

	const COLORS = array<vec3<f32>, 2>(
		vec3(1.0, 1.0, 1.0),
		vec3(0.0, 0.0, 0.0),
	);

    var out: VertexOutput;
	let model_position = vec3(x, y, z) + chunk.position;
    out.clip_position = camera.proj * camera.view * vec4(model_position, 1.0);
	out.world_position = model_position;
	out.world_normal = NORMALS[normal_index];
	out.color = COLORS[color_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let color = in.color;

	let ambient_brightness = 0.1;
	let ambient = light.color * ambient_brightness;

	let light_dir = normalize(light.position - in.world_position);
	let diffuse_brightness = max(dot(in.world_normal, light_dir), 0.0);
	let diffuse = light.color * diffuse_brightness;

	let result = (ambient + diffuse) * color;
    return vec4(result, 1.0);
}
