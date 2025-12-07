use crate::{
    camera::Camera,
    render::{
        byte_slice,
        chunk::{CHUNK_SIZE, Chunks},
    },
};
use glam::{Mat4, Quat, UVec3, Vec3};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

// Pre-shrinkification process memory usage sits at ~39MB.
// valgrind sits at roughly 13,626,576b heap allocation.
//
// Pose-shrinkification process memory sits at ~35.5MB.
// valgrind sits at roughly 27,962,600b heap allocation.
//
// I don't think I am using valgrind correctly lol.
//
// Seems to almost double performance? The fps measurement is
// not very scientific, but it seems like it went from ~450 to ~750.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VoxelVertex {
    /// color_index | normal_index | z | y | x
    /// ---------------------------------------
    /// 3             3              6   8   6
    packed: u32,
}

impl VoxelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32];

    const MAX_XZ: u32 = 63;
    const MAX_Y: u32 = 255;
    const Y_OFFSET: u32 = 6;
    const Z_OFFSET: u32 = 14;

    const NORMAL_OFFSET: u32 = 20;

    const MAX_COLOR_INDEX: u32 = 3;
    const COLOR_OFFSET: u32 = 23;
    const COLOR_MASK: u32 = 0x1800000;

    pub const fn new(translation: UVec3, normal: Vec3) -> Self {
        debug_assert!(translation.x <= Self::MAX_XZ);
        debug_assert!(translation.y <= Self::MAX_Y);
        debug_assert!(translation.z <= Self::MAX_XZ);

        let mut packed = 0;
        packed |= translation.x;
        packed |= translation.y << Self::Y_OFFSET;
        packed |= translation.z << Self::Z_OFFSET;

        let normal = match normal {
            Vec3::X => 0,
            Vec3::NEG_X => 1,
            Vec3::Y => 2,
            Vec3::NEG_Y => 3,
            Vec3::Z => 4,
            Vec3::NEG_Z => 5,
            _ => unreachable!(),
        };
        packed |= normal << Self::NORMAL_OFFSET;

        Self { packed }
    }

    pub fn offset(&mut self, translation: UVec3) {
        let packed = self.packed;
        let x = (packed & 0x3f) + translation.x;
        let y = ((packed >> Self::Y_OFFSET) & 0xff) + translation.y;
        let z = ((packed >> Self::Z_OFFSET) & 0x3f) + translation.z;

        debug_assert!(translation.x <= Self::MAX_XZ);
        debug_assert!(translation.y <= Self::MAX_Y);
        debug_assert!(translation.z <= Self::MAX_XZ);

        let remaining = packed & !0xFFFFF;
        self.packed = remaining
            | (x & 0x3f)
            | ((y & 0xff) << Self::Y_OFFSET)
            | ((z & 0x3f) << Self::Z_OFFSET);
    }

    pub fn set_color(&mut self, color: u32) {
        debug_assert!(color <= Self::MAX_COLOR_INDEX);
        let packed = self.packed & !Self::COLOR_MASK;
        self.packed = packed | (color << Self::COLOR_OFFSET);
    }

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// NOTE: Uniforms require 16 byte spacing
#[repr(C)]
#[derive(Clone, Copy)]
struct Light {
    position: Vec3,
    _pad: u32,
    color: [f32; 3],
    _pad2: u32,
}

#[repr(C)]
struct ChunkUniform {
    position: Vec3,
    _pad: u32,
}

pub struct VoxelPipeline {
    pipeline: wgpu::RenderPipeline,
    depth_buffer: wgpu::TextureView,
    chunks: Chunks,
    chunk_uniform: wgpu::Buffer,
    chunk_bind_group: wgpu::BindGroup,
    chunk_uniform_buffer: Vec<u8>,
    camera_uniform: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    light: Light,
    light_uniform: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    // Debug light source
    light_pipeline: wgpu::RenderPipeline,
    light_vertices: wgpu::Buffer,
    light_indices: wgpu::Buffer,
}

impl VoxelPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel camera uniform"),
            size: std::mem::size_of::<Mat4>() as wgpu::BufferAddress * 2,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform.as_entire_binding(),
            }],
        });

        let light = Light {
            position: Vec3::new(-30.0, 120.0, -80.0),
            _pad: 0,
            color: [1.0, 1.0, 1.0],
            _pad2: 0,
        };
        let light_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel light uniform"),
            contents: byte_slice(&[light]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("light bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light bind group"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_uniform.as_entire_binding(),
            }],
        });

        let light_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel vertex buffer"),
            contents: byte_slice(&VOXEL_FACES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices = VOXEL_FACES_INDICES
            .into_iter()
            .enumerate()
            .map(|(i, indices)| indices.map(|index| index + i as u32 * 4))
            .collect::<Vec<_>>();
        let light_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel index buffer"),
            contents: byte_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light pipeline layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("light shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/light.wgsl").into()),
        };
        let light_pipeline = crate::render::create_render_pipeline(
            device,
            &layout,
            surface_format,
            Some(wgpu::TextureFormat::Depth32Float),
            &[VoxelVertex::desc()],
            shader,
        );

        // TODO: The size of this buffer will depend on the devices `min_uniform_buffer_offset_alignment`.
        // Also, changing the number of chunks rendered would require resizing this buffer.
        let chunk_uniform_buffer =
            Vec::with_capacity(std::mem::size_of::<ChunkUniform>() * 4 * 4 * 256);
        let chunk_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk uniform"),
            size: std::mem::size_of::<ChunkUniform>() as wgpu::BufferAddress * 4 * 4 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let chunk_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("chunk bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let chunk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk bind group"),
            layout: &chunk_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &chunk_uniform,
                    offset: 0,
                    size: Some(
                        NonZeroU64::new(std::mem::size_of::<ChunkUniform>() as u64).unwrap(),
                    ),
                }),
            }],
        });

        let depth_buffer_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel depth buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_buffer =
            depth_buffer_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = wgpu::include_wgsl!("shaders/voxel.wgsl");
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel pipeline layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &chunk_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let pipeline = crate::render::create_render_pipeline(
            device,
            &render_pipeline_layout,
            surface_format,
            Some(wgpu::TextureFormat::Depth32Float),
            &[VoxelVertex::desc()],
            shader,
        );

        let chunks = Chunks::default();

        Self {
            pipeline,
            depth_buffer,
            chunks,
            chunk_uniform,
            chunk_bind_group,
            chunk_uniform_buffer,
            camera_uniform,
            camera_bind_group,
            light,
            light_uniform,
            light_bind_group,
            //
            light_pipeline,
            light_vertices,
            light_indices,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        let old_position = self.light.position;
        self.light.position =
            Quat::from_axis_angle((0.0, 1.0, 0.0).into(), 0.01f32.to_radians()) * old_position;
        queue.write_buffer(&self.light_uniform, 0, byte_slice(&[self.light]));

        let camera_matrices = [
            camera.projection_matrix(width, height),
            camera.view_matrix(),
        ];
        queue.write_buffer(&self.camera_uniform, 0, byte_slice(&camera_matrices));

        self.chunks.update(device, camera);

        queue.submit([]);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("debug light render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            render_pass.set_pipeline(&self.light_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.light_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.light_vertices.slice(..));
            render_pass.set_index_buffer(self.light_indices.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..36, 0, 0..1);
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_buffer,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_bind_group, &[]);

        let padding = device.limits().min_uniform_buffer_offset_alignment as usize
            - std::mem::size_of::<ChunkUniform>();

        self.chunk_uniform_buffer.clear();
        for (x, z) in self.chunks.0.keys() {
            self.chunk_uniform_buffer.extend(byte_slice(&[ChunkUniform {
                position: Vec3::new(
                    *x as f32 * CHUNK_SIZE as f32,
                    0.0,
                    *z as f32 * CHUNK_SIZE as f32,
                ),
                _pad: 0,
            }]));
            self.chunk_uniform_buffer.extend((0..padding).map(|_| 0));
        }

        queue.write_buffer(
            &self.chunk_uniform,
            0,
            byte_slice(&self.chunk_uniform_buffer),
        );

        let stride = device.limits().min_uniform_buffer_offset_alignment;
        for (i, chunk) in self.chunks.0.values().enumerate() {
            let offset = i as wgpu::DynamicOffset * stride;
            render_pass.set_bind_group(2, &self.chunk_bind_group, &[offset]);

            render_pass.set_vertex_buffer(0, chunk.vertices.slice(..));
            render_pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk.indices_count, 0, 0..1);
        }
    }
}

pub const VOXEL_FACES: [[VoxelVertex; 4]; 6] = [
    // Back face
    [
        VoxelVertex::new(UVec3::new(1, 0, 0), Vec3::NEG_Z),
        VoxelVertex::new(UVec3::new(1, 1, 0), Vec3::NEG_Z),
        VoxelVertex::new(UVec3::new(0, 0, 0), Vec3::NEG_Z),
        VoxelVertex::new(UVec3::new(0, 1, 0), Vec3::NEG_Z),
    ],
    // Front face
    [
        VoxelVertex::new(UVec3::new(1, 0, 1), Vec3::Z),
        VoxelVertex::new(UVec3::new(1, 1, 1), Vec3::Z),
        VoxelVertex::new(UVec3::new(0, 1, 1), Vec3::Z),
        VoxelVertex::new(UVec3::new(0, 0, 1), Vec3::Z),
    ],
    // Left face
    [
        VoxelVertex::new(UVec3::new(0, 1, 0), Vec3::NEG_X),
        VoxelVertex::new(UVec3::new(0, 0, 0), Vec3::NEG_X),
        VoxelVertex::new(UVec3::new(0, 0, 1), Vec3::NEG_X),
        VoxelVertex::new(UVec3::new(0, 1, 1), Vec3::NEG_X),
    ],
    // Right face
    [
        VoxelVertex::new(UVec3::new(1, 1, 0), Vec3::X),
        VoxelVertex::new(UVec3::new(1, 0, 0), Vec3::X),
        VoxelVertex::new(UVec3::new(1, 1, 1), Vec3::X),
        VoxelVertex::new(UVec3::new(1, 0, 1), Vec3::X),
    ],
    // Bottom face
    [
        VoxelVertex::new(UVec3::new(1, 0, 0), Vec3::NEG_Y),
        VoxelVertex::new(UVec3::new(1, 0, 1), Vec3::NEG_Y),
        VoxelVertex::new(UVec3::new(0, 0, 1), Vec3::NEG_Y),
        VoxelVertex::new(UVec3::new(0, 0, 0), Vec3::NEG_Y),
    ],
    // Top face
    [
        VoxelVertex::new(UVec3::new(1, 1, 0), Vec3::Y),
        VoxelVertex::new(UVec3::new(1, 1, 1), Vec3::Y),
        VoxelVertex::new(UVec3::new(0, 1, 0), Vec3::Y),
        VoxelVertex::new(UVec3::new(0, 1, 1), Vec3::Y),
    ],
];

pub const VOXEL_FACES_INDICES: [[u32; 6]; 6] = [
    // Back face
    [1, 0, 2, 1, 2, 3],
    // Front face
    [0, 1, 2, 0, 2, 3],
    // Left face
    [0, 1, 2, 0, 2, 3],
    // Right face
    [0, 2, 1, 2, 3, 1],
    // Bottom face
    [0, 1, 2, 0, 2, 3],
    // Top face
    [0, 2, 1, 2, 3, 1],
];
