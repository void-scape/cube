use crate::{
    camera::Camera,
    render::{
        byte_slice,
        camera::CameraData,
        vert::{VOXEL_FACES, VOXEL_FACES_INDICES, VoxelVertex},
    },
};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

// NOTE: Uniforms require 16 byte spacing
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LightUniform {
    pub proj_view: Mat4,
    pub position: Vec3,
    pub _pad: u32,
    pub color: [f32; 3],
    pub _pad2: u32,
}

pub struct LightData {
    pub light: LightUniform,
    pub uniform: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    // debug renderer
    pub pipeline: wgpu::RenderPipeline,
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
}

impl LightData {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_data: &CameraData,
    ) -> Self {
        let light = LightUniform {
            proj_view: Mat4::IDENTITY,
            position: Vec3::new(-100.0, 250.0, -100.0),
            _pad: 0,
            color: [1.0, 1.0, 1.0],
            _pad2: 0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel light uniform"),
            contents: byte_slice(&[light]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel vertex buffer"),
            contents: byte_slice(&VOXEL_FACES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices = VOXEL_FACES_INDICES
            .into_iter()
            .enumerate()
            .map(|(i, indices)| indices.map(|index| index + i as u32 * 4))
            .collect::<Vec<_>>();
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel index buffer"),
            contents: byte_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light pipeline layout"),
            bind_group_layouts: &[&camera_data.bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("light shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/debug_light.wgsl").into()),
        };
        let pipeline = crate::render::create_render_pipeline(
            device,
            &layout,
            Some(surface_format),
            Some(wgpu::TextureFormat::Depth32Float),
            &[VoxelVertex::desc()],
            shader,
        );

        Self {
            light,
            uniform,
            bind_group,
            bind_group_layout,
            pipeline,
            vertices,
            indices,
        }
    }

    pub fn prepare_render_pass(&mut self, queue: &wgpu::Queue, camera: &Camera) {
        // let old_position = self.light.position;
        // self.light.position =
        //     Quat::from_axis_angle((0.0, 1.0, 0.0).into(), 0.01f32.to_radians()) * old_position;
        self.light.proj_view = self.view_proj(camera);
        queue.write_buffer(&self.uniform, 0, byte_slice(&[self.light]));
    }

    pub fn debug_render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_buffer: &wgpu::TextureView,
        camera: &CameraData,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("debug light render pass"),
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
                view: depth_buffer,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &camera.bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertices.slice(..));
        render_pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..36, 0, 0..1);
    }

    fn view_proj(&self, _camera: &Camera) -> Mat4 {
        let size = 90.0;
        Mat4::orthographic_rh(-size, size, -size, size, 140.0, 350.0).mul_mat4(&Mat4::look_at_rh(
            self.light.position,
            Vec3::ZERO,
            Vec3::Y,
        ))
    }
}
