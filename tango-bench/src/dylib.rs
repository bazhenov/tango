use crate::MeasureTarget;
use libloading::{Library, Symbol};
use std::{
    ffi::{c_char, CStr},
    ptr::{addr_of, null_mut},
};

pub struct SharedObject<'l> {
    vt: ffi::VTable<'l>,
}

impl<'l> SharedObject<'l> {
    pub fn init(library: &'l Library) -> Self {
        let vt = ffi::VTable::new(library);
        unsafe {
            (vt.init)();
        }
        Self { vt }
    }

    pub fn count(&self) -> usize {
        unsafe { (self.vt.count)() }
    }

    pub fn select(&self, idx: usize) {
        unsafe { (self.vt.select)(idx) }
    }

    pub fn get_name(&self) -> &'l str {
        let mut length = 0usize;
        let name_ptr: *const c_char = null_mut();
        unsafe {
            (self.vt.get_name)(addr_of!(name_ptr) as _, &mut length);
            if length == 0 {
                return "";
            }
            let name = CStr::from_ptr(name_ptr).to_str().unwrap();
            &name[..length]
        }
    }

    pub fn run(&self) -> u64 {
        unsafe { (self.vt.run)() }
    }
}

/// State machine which implements the execution of FFI API (`tango_*` functions).
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
    use std::ptr::null;

    type InitFunc = unsafe extern "C" fn();
    type CountFunc = unsafe extern "C" fn() -> usize;
    type GetNameFunc = unsafe extern "C" fn(*mut *const c_char, *mut usize);
    type SelectFunc = unsafe extern "C" fn(usize);
    type RunFunc = unsafe extern "C" fn() -> u64;

    /// This block of constants is checking that all exported tango functions
    /// are of valid type according to the API. Those constants
    /// are not ment to be used at runtime in any way
    #[allow(unused)]
    mod type_check {
        use super::*;

        const TANGO_INIT: InitFunc = tango_init;
        const TANGO_COUNT: CountFunc = tango_count;
        const TANGO_SELECT: SelectFunc = tango_select;
        const TANGO_GET_NAME: GetNameFunc = tango_get_name;
        const TANGO_RUN: RunFunc = tango_run;
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

    pub(super) struct VTable<'l> {
        pub(super) init: Symbol<'l, InitFunc>,
        pub(super) count: Symbol<'l, CountFunc>,
        pub(super) select: Symbol<'l, SelectFunc>,
        pub(super) get_name: Symbol<'l, GetNameFunc>,
        pub(super) run: Symbol<'l, RunFunc>,
    }

    impl<'l> VTable<'l> {
        pub(super) fn new(library: &'l Library) -> Self {
            unsafe {
                let init = library
                    .get(b"tango_init\0")
                    .expect("Unable to get tango_init() symbol");

                let count = library
                    .get(b"tango_count\0")
                    .expect("Unable to get tango_count_functions() symbol");

                let select = library
                    .get(b"tango_select\0")
                    .expect("Unable to get tango_select() symbol");

                let get_name = library
                    .get(b"tango_get_name\0")
                    .expect("Unable to get tango_get_name() symbol");

                let run = library
                    .get(b"tango_run\0")
                    .expect("Unable to get tango_run() symbol");

                Self {
                    init,
                    count,
                    select,
                    get_name,
                    run,
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
