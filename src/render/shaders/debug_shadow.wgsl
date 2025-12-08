@group(0) @binding(0)
var shadow_map: texture_depth_2d; 
@group(0) @binding(1)
var shadow_map_sampler: sampler; 

const QUAD_SCALE: vec2<f32> = vec2(0.4, 0.4);
const QUAD_TRANSLATION: vec2<f32> = vec2(0.65, 0.65);

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let scaled = in.position * QUAD_SCALE + QUAD_TRANSLATION;
    out.clip_position = vec4<f32>(scaled, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(shadow_map, shadow_map_sampler, in.uv);
    return vec4(vec3(depth), 1.0);
}
