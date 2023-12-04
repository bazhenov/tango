//! Loading and resolving symbols from .dylib/.so libraries

use self::ffi::VTable;
use crate::{Error, MeasureTarget};
use libloading::{Library, Symbol};
use std::{
    ffi::c_char,
    ptr::{addr_of, null},
    slice, str,
};

pub struct Spi<'l> {
    tests: Vec<NamedFunction>,
    vt: Box<dyn VTable + 'l>,
}

pub struct NamedFunction {
    pub name: String,

    ///  Function index in FFI API
    idx: usize,
}

impl<'l> Spi<'l> {
    pub(crate) fn for_library(library: &'l Library) -> Result<Self, Error> {
        Self::for_vtable(ffi::LibraryVTable::new(library)?)
    }

    /// TODO: should be singleton
    pub(crate) fn for_self() -> Option<Result<Self, Error>> {
        unsafe { ffi::SELF_SPI.take().map(Self::for_vtable) }
    }

    fn for_vtable<T: VTable + 'l>(vt: T) -> Result<Self, Error> {
        let vt = Box::new(vt);
        vt.init();

        let mut tests = vec![];
        for idx in 0..vt.count() {
            vt.select(idx);

            let mut length = 0usize;
            let name_ptr: *const c_char = null();
            vt.get_test_name(addr_of!(name_ptr) as _, &mut length);
            if length == 0 {
                continue;
            }
            let slice = unsafe { slice::from_raw_parts(name_ptr as *const u8, length) };
            let name = str::from_utf8(slice)
                .map_err(Error::InvalidFFIString)?
                .to_string();
            tests.push(NamedFunction { name, idx });
        }

        Ok(Spi { vt, tests })
    }

    pub(crate) fn tests(&self) -> &[NamedFunction] {
        &self.tests
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<&NamedFunction> {
        self.tests.iter().find(|f| f.name == name)
    }

    pub(crate) fn run(&self, func: &NamedFunction, iterations: usize) -> u64 {
        self.vt.select(func.idx);
        self.vt.run(iterations)
    }

    pub(crate) fn estimate_iterations(&self, func: &NamedFunction, time_ms: u32) -> usize {
        self.vt.select(func.idx);
        self.vt.estimate_iterations(time_ms)
    }

    pub(crate) fn next_haystack(&self) -> bool {
        self.vt.next_haystack()
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

    /// Signature types of all FFI API functions
    type InitFn = unsafe extern "C" fn();
    type CountFn = unsafe extern "C" fn() -> usize;
    type GetTestNameFn = unsafe extern "C" fn(*mut *const c_char, *mut usize);
    type SelectFn = unsafe extern "C" fn(usize);
    type RunFn = unsafe extern "C" fn(usize) -> u64;
    type EstimateIterationsFn = unsafe extern "C" fn(u32) -> usize;
    type NextHaystackFn = unsafe extern "C" fn() -> bool;
    type FreeFn = unsafe extern "C" fn();

    /// This block of constants is checking that all exported tango functions are of valid type according to the API.
    /// Those constants are not ment to be used at runtime in any way
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
        /// Each benchmark executable should define this function for the harness to load all benchmarks
        fn __tango_create_benchmarks() -> Vec<Box<dyn MeasureTarget>>;
    }

    /// Global state of the benchmarking library
    static mut STATE: Option<State> = None;

    #[no_mangle]
    unsafe extern "C" fn tango_init() {
        if STATE.is_none() {
            STATE = Some(State {
                benchmarks: __tango_create_benchmarks(),
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
    unsafe extern "C" fn tango_next_haystack() -> bool {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().next_haystack()
        } else {
            false
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
        fn next_haystack(&self) -> bool;
    }

    pub(super) static mut SELF_SPI: Option<SelfVTable> = Some(SelfVTable);

    /// FFI implementation for the current executable.
    ///
    /// Used to communicate with FFI API of the executable bypassing dynamic linking.
    /// # Safety
    /// Instances of this type should not be created directory. The single instance [`SELF_SPI`] shoud be used instead
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

        fn next_haystack(&self) -> bool {
            unsafe { tango_next_haystack() }
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
        next_haystack_fn: Symbol<'l, NextHaystackFn>,
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

        fn next_haystack(&self) -> bool {
            unsafe { (self.next_haystack_fn)() }
        }
    }

    impl<'l> Drop for LibraryVTable<'l> {
        fn drop(&mut self) {
            unsafe { (self.free_fn)() }
        }
    }

    impl<'l> LibraryVTable<'l> {
        pub(super) fn new(library: &'l Library) -> Result<Self, Error> {
            unsafe {
                Ok(Self {
                    init_fn: lookup_symbol(library, "tango_init")?,
                    count_fn: lookup_symbol(library, "tango_count")?,
                    select_fn: lookup_symbol(library, "tango_select")?,
                    get_test_name_fn: lookup_symbol(library, "tango_get_test_name")?,
                    run_fn: lookup_symbol(library, "tango_run")?,
                    estimate_iterations_fn: lookup_symbol(library, "tango_estimate_iterations")?,
                    next_haystack_fn: lookup_symbol(library, "tango_next_haystack")?,
                    free_fn: lookup_symbol(library, "tango_free")?,
                })
            }
        }
    }

    unsafe fn lookup_symbol<'l, T>(
        library: &'l Library,
        name: &'static str,
    ) -> Result<Symbol<'l, T>, Error> {
        library
            .get(name.as_bytes())
            .map_err(Error::UnableToLoadSymbol)
    }
}
