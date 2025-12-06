use crate::platform::{PlatformInput, PlatformUpdate};

pub struct FnPtrs {
    #[allow(unused)]
    reloading: hot_reloading::HotReloading,
    handle_input: *mut core::ffi::c_void,
    update_and_render: *mut core::ffi::c_void,
}

impl FnPtrs {
    pub fn new<Memory>(
        handle_input: fn(PlatformInput<Memory>),
        update_and_render: fn(PlatformUpdate<Memory>),
        path: Option<&str>,
    ) -> Self {
        Self {
            reloading: hot_reloading::HotReloading::from_path(path),
            handle_input: handle_input as *mut core::ffi::c_void,
            update_and_render: update_and_render as *mut core::ffi::c_void,
        }
    }

    pub fn handle_input<Memory>(&self, input: PlatformInput<Memory>) {
        unsafe {
            let handle_input = core::mem::transmute::<
                *mut core::ffi::c_void,
                fn(PlatformInput<Memory>),
            >(self.handle_input);
            handle_input(input);
        }
    }

    pub fn update_and_render<Memory>(&self, input: PlatformUpdate<Memory>) {
        unsafe {
            let update_and_render = core::mem::transmute::<
                *mut core::ffi::c_void,
                fn(PlatformUpdate<Memory>),
            >(self.update_and_render);
            update_and_render(input);
        }
    }
}

#[cfg(not(feature = "hot-reload"))]
mod hot_reloading {
    use super::*;
    pub struct HotReloading;
    impl HotReloading {
        pub fn from_path(_: Option<&str>) -> Self {
            Self
        }
    }
    impl FnPtrs {
        pub fn reload(&mut self) -> bool {
            false
        }
    }
}

#[cfg(feature = "hot-reload")]
mod hot_reloading {
    use super::*;
    use std::ffi::CString;
    extern crate std;
    pub struct HotReloading {
        dylib: *mut core::ffi::c_void,
        path: Option<String>,
        loaded: std::time::SystemTime,
    }
    impl HotReloading {
        pub fn from_path(path: Option<&str>) -> Self {
            Self {
                dylib: core::ptr::null_mut(),
                path: path.map(|inner| inner.to_string()),
                loaded: std::time::SystemTime::now(),
            }
        }
    }
    impl FnPtrs {
        pub fn reload(&mut self) -> bool {
            let Some(path) = self.reloading.path.as_deref() else {
                return false;
            };
            let Some(modified) = std::fs::metadata(path).ok().and_then(|meta| {
                meta.modified().ok().and_then(|modified| {
                    modified
                        .duration_since(self.reloading.loaded)
                        .is_ok_and(|dur| !dur.is_zero())
                        .then_some(modified)
                })
            }) else {
                return false;
            };

            if !self.reloading.dylib.is_null() {
                // NOTE: This does nothing on macos.
                debug_assert_eq!(unsafe { libc::dlclose(self.reloading.dylib) }, 0);
            }
            self.reloading.loaded = modified;

            log::info!("loading functions from {path}");
            let mut copy = std::path::PathBuf::from(path);
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            copy.pop();
            copy.push(format!("{}", time.as_millis()));
            // NOTE: need to copy path on macos to prevent dylib caching
            std::fs::copy(path, &copy).expect("failed to copy dynamic library");
            let filename = CString::new(copy.to_str().unwrap()).unwrap();

            let dylib =
                unsafe { libc::dlopen(filename.as_ptr(), libc::RTLD_LOCAL | libc::RTLD_LAZY) };
            if !dylib.is_null() {
                let symbol = unsafe { libc::dlsym(dylib, c"update_and_render".as_ptr().cast()) };
                if !symbol.is_null() {
                    self.update_and_render = symbol;

                    let symbol = unsafe { libc::dlsym(dylib, c"handle_input".as_ptr().cast()) };
                    if !symbol.is_null() {
                        self.handle_input = symbol;
                    } else {
                        err("failed to load symbol handle_input");
                    }
                } else {
                    err("failed to load symbol update_and_render");
                }
            } else {
                err(&format!("failed to open {path}"));
            }

            fn err(msg: &str) {
                let str = unsafe { core::ffi::CStr::from_ptr(libc::dlerror()) };
                log::error!("{}: {}", msg, str.to_str().unwrap());
            }

            true
        }
    }
}
