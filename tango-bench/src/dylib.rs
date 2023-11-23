use self::ffi::Ffi;
use crate::MeasureTarget;
use libloading::{Library, Symbol};
use std::{
    ffi::{c_char, CStr},
    ptr::{addr_of, null_mut},
};

pub struct Spi<'l> {
    vtable: Box<dyn Ffi + 'l>,
}

impl<'l> Spi<'l> {
    pub fn for_library(library: &'l Library) -> Self {
        let vt = ffi::LibraryVTable::new(library);
        vt.init();
        Spi {
            vtable: Box::new(vt),
        }
    }

    pub fn for_self() -> Self {
        Spi {
            vtable: Box::new(ffi::SelfVTable),
        }
    }

    pub fn count(&self) -> usize {
        self.vtable.count()
    }

    pub fn select(&self, idx: usize) {
        self.vtable.select(idx)
    }

    pub fn get_name<'a>(&'a self) -> &'a str {
        let mut length = 0usize;
        let name_ptr: *const c_char = null_mut();
        self.vtable.get_name(addr_of!(name_ptr) as _, &mut length);
        if length == 0 {
            return "";
        }
        let name = unsafe { CStr::from_ptr(name_ptr) }.to_str().unwrap();
        &name[..length]
    }

    pub fn run(&self) -> u64 {
        self.vtable.run()
    }
}

/// State which holds the information about list of benchmarks and which one is selected.
/// Used in FFI API (`tango_*` functions).
struct State {
    benchmarks: Vec<Box<dyn MeasureTarget>>,
    selected_function: usize,
}

impl State {
    fn selected(&self) -> &dyn MeasureTarget {
        self.benchmarks[self.selected_function].as_ref()
    }

    fn selected_mut(&mut self) -> &mut dyn MeasureTarget {
        self.benchmarks[self.selected_function].as_mut()
    }
}

/// Defines all the foundation types and exported symbols for the FFI communication API between two
/// executables.
///
/// Tango execution model implies simultaneous exectution of the code from two binaries. To achive that
/// Tango benchmark is compiled in a way that executable is also a shared library (.dll, .so, .dylib). This
/// way two executables can coexist in the single process at the same time.
mod ffi {
    use super::*;
    use std::{os::raw::c_char, ptr::null, usize};

    type InitFn = unsafe extern "C" fn();
    type CountFn = unsafe extern "C" fn() -> usize;
    type GetNameFn = unsafe extern "C" fn(*mut *const c_char, *mut usize);
    type SelectFn = unsafe extern "C" fn(usize);
    type RunFn = unsafe extern "C" fn() -> u64;

    /// This block of constants is checking that all exported tango functions
    /// are of valid type according to the API. Those constants
    /// are not ment to be used at runtime in any way
    #[allow(unused)]
    mod type_check {
        use super::*;

        const TANGO_INIT: InitFn = tango_init;
        const TANGO_COUNT: CountFn = tango_count;
        const TANGO_SELECT: SelectFn = tango_select;
        const TANGO_GET_NAME: GetNameFn = tango_get_name;
        const TANGO_RUN: RunFn = tango_run;
    }

    extern "Rust" {
        fn create_benchmarks() -> Vec<Box<dyn MeasureTarget>>;
    }

    /// Global state of the benchmarking library
    static mut STATE: Option<State> = None;

    #[no_mangle]
    unsafe extern "C" fn tango_init() {
        if STATE.is_none() {
            STATE = Some(State {
                benchmarks: create_benchmarks(),
                selected_function: 0,
            });
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_count() -> usize {
        STATE.as_ref().map(|s| s.benchmarks.len()).unwrap_or(0)
    }

    #[no_mangle]
    unsafe extern "C" fn tango_select(idx: usize) {
        if let Some(s) = STATE.as_mut() {
            s.selected_function = idx.min(s.benchmarks.len() - 1);
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_free() {
        STATE.take();
    }

    #[no_mangle]
    unsafe extern "C" fn tango_get_name(name: *mut *const c_char, length: *mut usize) {
        if let Some(s) = STATE.as_ref() {
            let n = s.selected().name();
            *name = n.as_ptr() as _;
            *length = n.len();
        } else {
            *name = null();
            *length = 0;
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_run() -> u64 {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().measure(10)
        } else {
            0
        }
    }

    pub trait Ffi {
        fn init(&self);
        fn count(&self) -> usize;
        fn select(&self, func_idx: usize);
        fn get_name(&self, ptr: *mut *const c_char, len: *mut usize);
        fn run(&self) -> u64;
    }

    pub struct SelfVTable;

    impl Ffi for SelfVTable {
        fn init(&self) {
            unsafe { tango_init() }
        }

        fn count(&self) -> usize {
            unsafe { tango_count() }
        }

        fn select(&self, func_idx: usize) {
            unsafe { tango_select(func_idx) }
        }

        fn get_name(&self, ptr: *mut *const c_char, len: *mut usize) {
            unsafe { tango_get_name(ptr, len) }
        }

        fn run(&self) -> u64 {
            unsafe { tango_run() }
        }
    }

    pub struct LibraryVTable<'l> {
        pub(super) init_fn: Symbol<'l, InitFn>,
        pub(super) count_fn: Symbol<'l, CountFn>,
        pub(super) select_fn: Symbol<'l, SelectFn>,
        pub(super) get_name_fn: Symbol<'l, GetNameFn>,
        pub(super) run_fn: Symbol<'l, RunFn>,
    }

    impl<'l> Ffi for LibraryVTable<'l> {
        fn init(&self) {
            unsafe { (self.init_fn)() }
        }

        fn count(&self) -> usize {
            unsafe { (self.count_fn)() }
        }

        fn select(&self, func_idx: usize) {
            unsafe { (self.select_fn)(func_idx) }
        }

        fn get_name(&self, ptr: *mut *const c_char, len: *mut usize) {
            unsafe { (self.get_name_fn)(ptr, len) }
        }

        fn run(&self) -> u64 {
            unsafe { (self.run_fn)() }
        }
    }

    impl<'l> LibraryVTable<'l> {
        pub(super) fn new(library: &'l Library) -> Self {
            unsafe {
                let init_fn = library
                    .get(b"tango_init\0")
                    .expect("Unable to get tango_init() symbol");

                let count_fn = library
                    .get(b"tango_count\0")
                    .expect("Unable to get tango_count_functions() symbol");

                let select_fn = library
                    .get(b"tango_select\0")
                    .expect("Unable to get tango_select() symbol");

                let get_name_fn = library
                    .get(b"tango_get_name\0")
                    .expect("Unable to get tango_get_name() symbol");

                let run_fn = library
                    .get(b"tango_run\0")
                    .expect("Unable to get tango_run() symbol");

                Self {
                    init_fn,
                    count_fn,
                    select_fn,
                    get_name_fn,
                    run_fn,
                }
            }
        }
    }
}

#[deprecated]
#[macro_export]
macro_rules! prevent_shared_function_deletion {
    () => {
        use tango_bench::cli::dylib;
        let funcs: &[*const fn()] = &[dylib::tango_init as _];
        #[allow(forgetting_references)]
        std::mem::forget(funcs);
    };
}
