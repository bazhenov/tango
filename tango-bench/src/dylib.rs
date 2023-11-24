use self::ffi::{SelfVTable, VTable};
use crate::MeasureTarget;
use core::slice;
use libloading::{Library, Symbol};
use std::{
    collections::BTreeMap,
    ffi::c_char,
    ptr::{addr_of, null},
    str,
};

static mut SELF_SPI: Option<SelfVTable> = Some(SelfVTable);

pub struct Spi<'l> {
    tests: BTreeMap<String, usize>,
    vt: Box<dyn VTable + 'l>,
}

impl<'l> Spi<'l> {
    pub fn for_library(library: &'l Library) -> Self {
        Self::for_vtable(ffi::LibraryVTable::new(library))
    }

    /// TODO: should be singleton
    pub fn for_self() -> Option<Self> {
        unsafe { SELF_SPI.take().map(Self::for_vtable) }
    }

    fn for_vtable<T: VTable + 'l>(vt: T) -> Self {
        let vt = Box::new(vt);
        vt.init();

        let mut tests = BTreeMap::new();
        for i in 0..vt.count() {
            vt.select(i);

            let mut length = 0usize;
            let name_ptr: *const c_char = null();
            vt.get_test_name(addr_of!(name_ptr) as _, &mut length);
            if length == 0 {
                continue;
            }
            let slice = unsafe { slice::from_raw_parts(name_ptr as *const u8, length) };
            let str = str::from_utf8(slice).unwrap();
            tests.insert(str.to_string(), i);
        }

        Spi { vt, tests }
    }

    pub fn tests(&self) -> &BTreeMap<String, usize> {
        &self.tests
    }

    pub fn run(&self, idx: usize, iterations: usize) -> u64 {
        self.vt.select(idx);
        self.vt.run(iterations)
    }

    pub fn estimate_iterations(&self, idx: usize, time_ms: u32) -> usize {
        self.vt.select(idx);
        self.vt.estimate_iterations(time_ms)
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
    type GetTestNameFn = unsafe extern "C" fn(*mut *const c_char, *mut usize);
    type SelectFn = unsafe extern "C" fn(usize);
    type RunFn = unsafe extern "C" fn(usize) -> u64;
    type EstimateIterationsFn = unsafe extern "C" fn(u32) -> usize;
    type FreeFn = unsafe extern "C" fn();

    /// This block of constants is checking that all exported tango functions
    /// are of valid type according to the API. Those constants
    /// are not ment to be used at runtime in any way
    #[allow(unused)]
    mod type_check {
        use super::*;

        const TANGO_INIT: InitFn = tango_init;
        const TANGO_COUNT: CountFn = tango_count;
        const TANGO_SELECT: SelectFn = tango_select;
        const TANGO_GET_TEST_NAME: GetTestNameFn = tango_get_test_name;
        const TANGO_RUN: RunFn = tango_run;
        const TANGO_ESTIMATE_ITERATIONS: EstimateIterationsFn = tango_estimate_iterations;
        const TANGO_FREE: FreeFn = tango_free;
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
    unsafe extern "C" fn tango_get_test_name(name: *mut *const c_char, length: *mut usize) {
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
    unsafe extern "C" fn tango_run(iterations: usize) -> u64 {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().measure(iterations)
        } else {
            0
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_estimate_iterations(time_ms: u32) -> usize {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().estimate_iterations(time_ms)
        } else {
            0
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_free() {
        STATE.take();
    }

    pub(super) trait VTable {
        fn init(&self);
        fn count(&self) -> usize;
        fn select(&self, func_idx: usize);
        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut usize);
        fn run(&self, iterations: usize) -> u64;
        fn estimate_iterations(&self, time_ms: u32) -> usize;
    }

    pub(super) struct SelfVTable;

    impl VTable for SelfVTable {
        fn init(&self) {
            unsafe { tango_init() }
        }

        fn count(&self) -> usize {
            unsafe { tango_count() }
        }

        fn select(&self, func_idx: usize) {
            unsafe { tango_select(func_idx) }
        }

        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut usize) {
            unsafe { tango_get_test_name(ptr, len) }
        }

        fn run(&self, iterations: usize) -> u64 {
            unsafe { tango_run(iterations) }
        }

        fn estimate_iterations(&self, time_ms: u32) -> usize {
            unsafe { tango_estimate_iterations(time_ms) }
        }
    }

    impl Drop for SelfVTable {
        fn drop(&mut self) {
            unsafe {
                tango_free();
            }
        }
    }

    pub(super) struct LibraryVTable<'l> {
        init_fn: Symbol<'l, InitFn>,
        count_fn: Symbol<'l, CountFn>,
        select_fn: Symbol<'l, SelectFn>,
        get_test_name_fn: Symbol<'l, GetTestNameFn>,
        run_fn: Symbol<'l, RunFn>,
        estimate_iterations_fn: Symbol<'l, EstimateIterationsFn>,
        free_fn: Symbol<'l, FreeFn>,
    }

    impl<'l> VTable for LibraryVTable<'l> {
        fn init(&self) {
            unsafe { (self.init_fn)() }
        }

        fn count(&self) -> usize {
            unsafe { (self.count_fn)() }
        }

        fn select(&self, func_idx: usize) {
            unsafe { (self.select_fn)(func_idx) }
        }

        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut usize) {
            unsafe { (self.get_test_name_fn)(ptr, len) }
        }

        fn run(&self, iterations: usize) -> u64 {
            unsafe { (self.run_fn)(iterations) }
        }

        fn estimate_iterations(&self, time_ms: u32) -> usize {
            unsafe { (self.estimate_iterations_fn)(time_ms) }
        }
    }

    impl<'l> Drop for LibraryVTable<'l> {
        fn drop(&mut self) {
            unsafe { (self.free_fn)() }
        }
    }

    impl<'l> LibraryVTable<'l> {
        pub(super) fn new(library: &'l Library) -> Self {
            unsafe {
                Self {
                    init_fn: lookup_symbol(library, "tango_init"),
                    count_fn: lookup_symbol(library, "tango_count"),
                    select_fn: lookup_symbol(library, "tango_select"),
                    get_test_name_fn: lookup_symbol(library, "tango_get_test_name"),
                    run_fn: lookup_symbol(library, "tango_run"),
                    estimate_iterations_fn: lookup_symbol(library, "tango_estimate_iterations"),
                    free_fn: lookup_symbol(library, "tango_free"),
                }
            }
        }
    }

    unsafe fn lookup_symbol<'l, T>(library: &'l Library, name: &'static str) -> Symbol<'l, T> {
        library
            .get(name.as_bytes())
            .unwrap_or_else(|_| panic!("Unable to get {}() symbol", name))
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
