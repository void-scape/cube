use crate::{
    camera::Camera,
    render::{camera::CameraData, chunk::ChunkData, light::LightData, shadow::ShadowData},
};

pub struct VoxelPipeline {
    depth_buffer: wgpu::TextureView,
    chunks: ChunkData,
    camera: CameraData,
    light: LightData,
    shadow: ShadowData,
}

impl VoxelPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
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

        let camera = CameraData::new(device);
        let light = LightData::new(device, surface_format, &camera);

        let shadow_map_bind_group_layout = ShadowData::shadow_map_bind_group_layout(device);
        let chunks = ChunkData::new(
            device,
            surface_format,
            &camera,
            &light,
            &shadow_map_bind_group_layout,
        );
        let shadow = ShadowData::new(
            device,
            shadow_map_bind_group_layout,
            surface_format,
            &chunks,
        );

        Self {
            depth_buffer,
            chunks,
            camera,
            light,
            shadow,
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
        self.camera.update(queue, width, height, camera);
        self.chunks.update(device, camera);

        self.light.prepare_render_pass(queue, camera);
        // self.shadow.prepare_render_pass(queue, &self.light);
        self.chunks.render(
            device,
            queue,
            encoder,
            view,
            &self.depth_buffer,
            &self.camera,
            &self.light,
            &self.shadow,
        );
        self.light
            .debug_render(encoder, view, &self.depth_buffer, &self.camera);
        // self.shadow.debug_render(encoder, view);
    }
}
