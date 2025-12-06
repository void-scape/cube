use callback::FnPtrs;
use core::num::NonZeroU32;
use std::sync::Arc;
use time::Time;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

mod callback;
mod time;

pub fn run<Memory>(
    mem: Memory,
    width: usize,
    height: usize,
    handle_input: fn(PlatformInput<Memory>),
    update_and_render: fn(PlatformUpdate<Memory>),
    reload: Option<&str>,
) where
    Memory: 'static,
{
    #[allow(unused_mut)]
    let mut app = App {
        window: None,
        gfx: None,
        //
        width: width as u32,
        height: height as u32,
        mem,
        now: Time::now(),
        fns: FnPtrs::new(handle_input, update_and_render, reload),
    };

    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    event_loop.run_app(&mut app).unwrap();
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    {
        console_error_panic_hook::set_once();
        winit::platform::web::EventLoopExtWebSys::spawn_app(event_loop, app);
    }
}

struct App<Memory> {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    //
    width: u32,
    height: u32,
    mem: Memory,
    now: Time,
    fns: FnPtrs,
}

// https://github.com/gfx-rs/wgpu/blob/trunk/examples/standalone/02_hello_window/src/main.rs
struct Gfx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
}

impl Gfx {
    async fn new(window: Arc<Window>, width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let gfx = Self {
            device,
            queue,
            surface,
            surface_format,
        };

        // Configure surface for the first time
        gfx.configure_surface(width, height);

        gfx
    }

    fn configure_surface(&self, width: u32, height: u32) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width,
            height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoNoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.configure_surface(width, height);
    }
}

impl<Memory> ApplicationHandler for App<Memory> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(window_attributes(self.width, self.height))
                .unwrap(),
        );
        let gfx = pollster::block_on(Gfx::new(window.clone(), self.width, self.height));
        self.gfx = Some(gfx);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.as_mut()
                    && let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                {
                    self.width = w.into();
                    self.height = h.into();
                    gfx.resize(self.width, self.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let (Some(window), Some(gfx)) = (&self.window, &mut self.gfx) else {
                    return;
                };

                let reloaded = self.fns.reload();

                let delta = {
                    let now = Time::now();
                    let delta = now.elapsed_secs(self.now);
                    self.now = now;
                    delta
                };

                let surface_texture = gfx
                    .surface
                    .get_current_texture()
                    .expect("failed to acquire next swapchain texture");

                self.fns.update_and_render(PlatformUpdate {
                    memory: &mut self.mem,
                    delta,
                    //
                    window,
                    event_loop,
                    //
                    width: self.width,
                    height: self.height,
                    queue: &gfx.queue,
                    device: &gfx.device,
                    surface_texture: &surface_texture,
                    surface_format: gfx.surface_format,
                    //
                    reloaded,
                });

                window.pre_present_notify();
                surface_texture.present();
                window.request_redraw();
            }
            _ => {}
        }

        let Some(window) = &self.window else {
            return;
        };

        self.fns.handle_input(PlatformInput {
            memory: &mut self.mem,
            window,
            input: Input::Window(event),
        });
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let Some(window) = &self.window else {
            return;
        };

        self.fns.handle_input(PlatformInput {
            memory: &mut self.mem,
            window,
            input: Input::Device(event),
        });
    }
}

fn window_attributes(width: u32, height: u32) -> WindowAttributes {
    let attributes = Window::default_attributes().with_inner_size(PhysicalSize::new(width, height));
    #[cfg(target_arch = "wasm32")]
    let attributes = winit::platform::web::WindowAttributesExtWebSys::with_append(attributes, true);
    attributes
}

pub struct PlatformUpdate<'a, T> {
    // logic
    pub memory: &'a mut T,
    pub delta: f32,

    // window
    pub event_loop: &'a winit::event_loop::ActiveEventLoop,
    pub window: &'a winit::window::Window,

    // graphics
    pub width: u32,
    pub height: u32,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface_texture: &'a wgpu::SurfaceTexture,
    pub surface_format: wgpu::TextureFormat,

    // debug
    pub reloaded: bool,
}

pub struct PlatformInput<'a, T> {
    pub memory: &'a mut T,
    pub window: &'a winit::window::Window,
    pub input: Input,
}

pub enum Input {
    Window(winit::event::WindowEvent),
    Device(winit::event::DeviceEvent),
}

// Debug utility

pub use debug::{
    debug_target, debug_time_micros, debug_time_millis, debug_time_nanos, debug_time_secs,
};

pub mod debug {
    /// Automatically generate a path to the crate's dynamic library in `target/debug`.
    ///
    /// Returns `None` if `debug_assertions` are disabled.
    pub fn debug_target() -> Option<&'static str> {
        #[cfg(all(debug_assertions, any(target_os = "linux", target_os = "macos")))]
        {
            #[cfg(target_os = "linux")]
            let extension = "so";
            #[cfg(target_os = "macos")]
            let extension = "dylib";

            let name = env!("CARGO_CRATE_NAME");
            let path = format!("target/debug/lib{}.{}", name, extension);
            return match std::fs::exists(&path) {
                Ok(_) => Some(std::string::String::leak(path)),
                Err(err) => panic!("failed to load {path}: {err}"),
            };
        }
        #[allow(unused)]
        None
    }

    pub fn debug_time_secs<R>(mut f: impl FnMut() -> R) -> (f32, R) {
        let start = std::time::Instant::now();
        let result = f();
        let duration = std::time::Instant::now()
            .duration_since(start)
            .as_secs_f32();
        (duration, result)
    }

    pub fn debug_time_millis<R>(mut f: impl FnMut() -> R) -> (u128, R) {
        let start = std::time::Instant::now();
        let result = f();
        let duration = std::time::Instant::now().duration_since(start).as_millis();
        (duration, result)
    }

    pub fn debug_time_micros<R>(mut f: impl FnMut() -> R) -> (u128, R) {
        let start = std::time::Instant::now();
        let result = f();
        let duration = std::time::Instant::now().duration_since(start).as_micros();
        (duration, result)
    }

    pub fn debug_time_nanos<R>(mut f: impl FnMut() -> R) -> (u128, R) {
        let start = std::time::Instant::now();
        let result = f();
        let duration = std::time::Instant::now().duration_since(start).as_nanos();
        (duration, result)
    }
}
