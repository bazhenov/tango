//! Loading and resolving symbols from .dylib/.so libraries

use self::ffi::{VTable, SELF_VTABLE};
use crate::{Benchmark, Error};
use anyhow::Context;
use libloading::{Library, Symbol};
use std::{
    ffi::{c_char, c_ulong},
    path::Path,
    ptr::{addr_of, null},
    slice, str,
    sync::mpsc::{channel, Receiver, Sender},
    thread::{self, JoinHandle},
};

pub type FunctionIdx = usize;

#[derive(Debug, Clone)]
pub struct NamedFunction {
    pub name: String,

    /// Function index in FFI API
    pub idx: FunctionIdx,
}

pub(crate) struct Spi {
    tests: Vec<NamedFunction>,
    selected_function: Option<FunctionIdx>,
    mode: SpiMode,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SpiModeKind {
    Synchronous,
    Asynchronous,
}

enum SpiMode {
    Synchronous {
        vt: Box<dyn VTable>,
        last_measurement: u64,
    },
    Asynchronous {
        worker: Option<JoinHandle<()>>,
        tx: Sender<SpiRequest>,
        rx: Receiver<SpiReply>,
    },
}

impl Spi {
    pub(crate) fn for_library(path: impl AsRef<Path>, mode: SpiModeKind) -> Spi {
        let lib = unsafe { Library::new(path.as_ref()) }
            .with_context(|| format!("Unable to open library: {}", path.as_ref().display()))
            .unwrap();
        spi_handle_for_vtable(ffi::LibraryVTable::new(lib).unwrap(), mode)
    }

    pub(crate) fn for_self(mode: SpiModeKind) -> Option<Spi> {
        unsafe { SELF_VTABLE.take() }.map(|vt| spi_handle_for_vtable(vt, mode))
    }

    pub(crate) fn tests(&self) -> &[NamedFunction] {
        &self.tests
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<&NamedFunction> {
        self.tests.iter().find(|f| f.name == name)
    }

    pub(crate) fn run(&mut self, func: FunctionIdx, iterations: usize) -> u64 {
        self.select(func);
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.run(iterations as c_ulong),
            SpiMode::Asynchronous { worker: _, tx, rx } => {
                tx.send(SpiRequest::Run { iterations }).unwrap();
                match rx.recv().unwrap() {
                    SpiReply::Run(time) => time,
                    r => panic!("Unexpected response: {:?}", r),
                }
            }
        }
    }

    pub(crate) fn measure(&mut self, func: FunctionIdx, iterations: usize) {
        self.select(func);
        match &mut self.mode {
            SpiMode::Synchronous {
                vt,
                last_measurement,
            } => {
                *last_measurement = vt.run(iterations as c_ulong);
            }
            SpiMode::Asynchronous { tx, .. } => {
                tx.send(SpiRequest::Measure { iterations }).unwrap();
            }
        }
    }

    pub(crate) fn read_sample(&mut self) -> u64 {
        match &self.mode {
            SpiMode::Synchronous {
                last_measurement, ..
            } => *last_measurement,
            SpiMode::Asynchronous { rx, .. } => match rx.recv().unwrap() {
                SpiReply::Measure(time) => time,
                r => panic!("Unexpected response: {:?}", r),
            },
        }
    }

    pub(crate) fn estimate_iterations(&mut self, func: FunctionIdx, time_ms: u32) -> usize {
        self.select(func);
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.estimate_iterations(time_ms) as usize,
            SpiMode::Asynchronous { tx, rx, .. } => {
                tx.send(SpiRequest::EstimateIterations { time_ms }).unwrap();
                match rx.recv().unwrap() {
                    SpiReply::EstimateIterations(iters) => iters,
                    r => panic!("Unexpected response: {:?}", r),
                }
            }
        }
    }

    pub(crate) fn prepare_state(&mut self, func: FunctionIdx, seed: u64) -> bool {
        self.select(func);
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.prepare_state(seed),
            SpiMode::Asynchronous { tx, rx, .. } => {
                tx.send(SpiRequest::PrepareState { seed }).unwrap();
                match rx.recv().unwrap() {
                    SpiReply::PrepareState(ret) => ret,
                    r => panic!("Unexpected response: {:?}", r),
                }
            }
        }
    }

    pub(crate) fn select(&mut self, idx: usize) {
        let different_function = self.selected_function.map(|v| v != idx).unwrap_or(true);
        if different_function {
            match &self.mode {
                SpiMode::Synchronous { vt, .. } => vt.select(idx as c_ulong),
                SpiMode::Asynchronous { tx, rx, .. } => {
                    tx.send(SpiRequest::Select { idx }).unwrap();
                    match rx.recv().unwrap() {
                        SpiReply::Select => self.selected_function = Some(idx),
                        r => panic!("Unexpected response: {:?}", r),
                    }
                }
            }
        }
    }
}

impl Drop for Spi {
    fn drop(&mut self) {
        if let SpiMode::Asynchronous { worker, tx, .. } = &mut self.mode {
            if let Some(worker) = worker.take() {
                tx.send(SpiRequest::Shutdown).unwrap();
                worker.join().unwrap();
            }
        }
    }
}

fn spi_worker(vt: &dyn VTable, rx: Receiver<SpiRequest>, tx: Sender<SpiReply>) {
    use SpiReply as Rp;
    use SpiRequest as Rq;

    while let Ok(req) = rx.recv() {
        let reply = match req {
            Rq::EstimateIterations { time_ms } => {
                Rp::EstimateIterations(vt.estimate_iterations(time_ms) as usize)
            }
            Rq::PrepareState { seed } => Rp::PrepareState(vt.prepare_state(seed)),
            Rq::Select { idx } => {
                vt.select(idx as c_ulong);
                Rp::Select
            }
            Rq::Run { iterations } => Rp::Run(vt.run(iterations as c_ulong)),
            Rq::Measure { iterations } => Rp::Measure(vt.run(iterations as c_ulong)),
            Rq::Shutdown => break,
        };
        tx.send(reply).unwrap();
    }
}

fn spi_handle_for_vtable(vtable: impl VTable + Send + 'static, mode: SpiModeKind) -> Spi {
    vtable.init();
    let tests = enumerate_tests(&vtable).unwrap();

    match mode {
        SpiModeKind::Asynchronous => {
            let (request_tx, request_rx) = channel();
            let (reply_tx, reply_rx) = channel();
            let worker = thread::spawn(move || {
                spi_worker(&vtable, request_rx, reply_tx);
            });

            Spi {
                tests,
                selected_function: None,
                mode: SpiMode::Asynchronous {
                    worker: Some(worker),
                    tx: request_tx,
                    rx: reply_rx,
                },
            }
        }
        SpiModeKind::Synchronous => Spi {
            tests,
            selected_function: None,
            mode: SpiMode::Synchronous {
                vt: Box::new(vtable),
                last_measurement: 0,
            },
        },
    }
}

fn enumerate_tests(vt: &dyn VTable) -> Result<Vec<NamedFunction>, Error> {
    let mut tests = vec![];
    for idx in 0..vt.count() {
        vt.select(idx);

        let mut length = 0;
        let name_ptr: *const c_char = null();
        vt.get_test_name(addr_of!(name_ptr) as _, &mut length);
        if length == 0 {
            continue;
        }
        let slice = unsafe { slice::from_raw_parts(name_ptr as *const u8, length as usize) };
        let name = str::from_utf8(slice)
            .map_err(Error::InvalidFFIString)?
            .to_string();
        let idx = idx as usize;
        tests.push(NamedFunction { name, idx });
    }
    Ok(tests)
}

enum SpiRequest {
    EstimateIterations { time_ms: u32 },
    PrepareState { seed: u64 },
    Select { idx: usize },
    Run { iterations: usize },
    Measure { iterations: usize },
    Shutdown,
}

#[derive(Debug)]
enum SpiReply {
    EstimateIterations(usize),
    PrepareState(bool),
    Select,
    Run(u64),
    Measure(u64),
}

/// State which holds the information about list of benchmarks and which one is selected.
/// Used in FFI API (`tango_*` functions).
pub struct State {
    pub benchmarks: Vec<Benchmark>,
    pub selected_function: usize,
}

impl State {
    fn selected(&self) -> &Benchmark {
        &self.benchmarks[self.selected_function]
    }

    fn selected_mut(&mut self) -> &mut Benchmark {
        &mut self.benchmarks[self.selected_function]
    }
}

/// Global state of the benchmarking library
static mut STATE: Option<State> = None;

/// `tango_init()` implementation
///
/// This function is not exported from the library, but is used by the `tango_init()` functions
/// generated by the `tango_benchmark!()` macro.
pub fn __tango_init(benchmarks: Vec<Benchmark>) {
    unsafe {
        if STATE.is_none() {
            STATE = Some(State {
                benchmarks,
                selected_function: 0,
            });
        }
    }
}

/// Defines all the foundation types and exported symbols for the FFI communication API between two
/// executables.
///
/// Tango execution model implies simultaneous exectution of the code from two binaries. To achive that
/// Tango benchmark is compiled in a way that executable is also a shared library (.dll, .so, .dylib). This
/// way two executables can coexist in the single process at the same time.
pub mod ffi {
    use super::*;
    use std::{
        ffi::{c_uint, c_ulong},
        mem,
        os::raw::c_char,
        ptr::null,
        usize,
    };

    /// Signature types of all FFI API functions
    pub type InitFn = unsafe extern "C" fn();
    type CountFn = unsafe extern "C" fn() -> c_ulong;
    type GetTestNameFn = unsafe extern "C" fn(*mut *const c_char, *mut c_ulong);
    type SelectFn = unsafe extern "C" fn(c_ulong);
    type RunFn = unsafe extern "C" fn(c_ulong) -> u64;
    type EstimateIterationsFn = unsafe extern "C" fn(c_uint) -> c_ulong;
    type PrepareStateFn = unsafe extern "C" fn(c_ulong) -> bool;
    type FreeFn = unsafe extern "C" fn();

    /// This block of constants is checking that all exported tango functions are of valid type according to the API.
    /// Those constants are not ment to be used at runtime in any way
    #[allow(unused)]
    mod type_check {
        use super::*;

        const TANGO_COUNT: CountFn = tango_count;
        const TANGO_SELECT: SelectFn = tango_select;
        const TANGO_GET_TEST_NAME: GetTestNameFn = tango_get_test_name;
        const TANGO_RUN: RunFn = tango_run;
        const TANGO_ESTIMATE_ITERATIONS: EstimateIterationsFn = tango_estimate_iterations;
        const TANGO_FREE: FreeFn = tango_free;
    }

    #[no_mangle]
    unsafe extern "C" fn tango_count() -> c_ulong {
        STATE
            .as_ref()
            .map(|s| s.benchmarks.len() as c_ulong)
            .unwrap_or(0)
    }

    #[no_mangle]
    unsafe extern "C" fn tango_select(idx: c_ulong) {
        if let Some(s) = STATE.as_mut() {
            s.selected_function = idx.min((s.benchmarks.len() - 1) as c_ulong) as usize;
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_get_test_name(name: *mut *const c_char, length: *mut c_ulong) {
        if let Some(s) = STATE.as_ref() {
            let n = s.selected().name();
            *name = n.as_ptr() as _;
            *length = n.len() as c_ulong;
        } else {
            *name = null();
            *length = 0;
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_run(iterations: c_ulong) -> u64 {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().measure(iterations as usize)
        } else {
            0
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_estimate_iterations(time_ms: c_uint) -> c_ulong {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().estimate_iterations(time_ms) as c_ulong
        } else {
            0
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_prepare_state(seed: c_ulong) -> bool {
        if let Some(s) = STATE.as_mut() {
            s.selected_mut().prepare_state(seed)
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
        fn count(&self) -> c_ulong;
        fn select(&self, func_idx: c_ulong);
        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut c_ulong);
        fn run(&self, iterations: c_ulong) -> c_ulong;
        fn estimate_iterations(&self, time_ms: c_uint) -> c_ulong;
        fn prepare_state(&self, seed: c_ulong) -> bool;
    }

    pub(super) static mut SELF_VTABLE: Option<SelfVTable> = Some(SelfVTable);

    /// FFI implementation for the current executable.
    ///
    /// Used to communicate with FFI API of the executable bypassing dynamic linking.
    /// # Safety
    /// Instances of this type should not be created directory. The single instance [`SELF_SPI`] shoud be used instead
    pub(super) struct SelfVTable;

    impl VTable for SelfVTable {
        fn init(&self) {
            // In executable mode `tango_init` is already called by the main function
        }

        fn count(&self) -> c_ulong {
            unsafe { tango_count() }
        }

        fn select(&self, func_idx: c_ulong) {
            unsafe { tango_select(func_idx) }
        }

        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut c_ulong) {
            unsafe { tango_get_test_name(ptr, len) }
        }

        fn run(&self, iterations: c_ulong) -> u64 {
            unsafe { tango_run(iterations) }
        }

        fn estimate_iterations(&self, time_ms: c_uint) -> c_ulong {
            unsafe { tango_estimate_iterations(time_ms) }
        }

        fn prepare_state(&self, seed: u64) -> bool {
            unsafe { tango_prepare_state(seed) }
        }
    }

    impl Drop for SelfVTable {
        fn drop(&mut self) {
            unsafe {
                tango_free();
            }
        }
    }

    pub(super) struct LibraryVTable {
        /// SAFETY: using static here is sound because
        ///  (1) this struct is private and field can not be accessed outside
        ///  (2) rust has drop order guarantee (fields are dropped in declaration order)
        init_fn: Symbol<'static, InitFn>,
        count_fn: Symbol<'static, CountFn>,
        select_fn: Symbol<'static, SelectFn>,
        get_test_name_fn: Symbol<'static, GetTestNameFn>,
        run_fn: Symbol<'static, RunFn>,
        estimate_iterations_fn: Symbol<'static, EstimateIterationsFn>,
        prepare_state_fn: Symbol<'static, PrepareStateFn>,
        free_fn: Symbol<'static, FreeFn>,

        /// SAFETY: This field should be last because it should be dropped last
        _library: Box<Library>,
    }

    impl LibraryVTable {
        pub(super) fn new(library: Library) -> Result<Self, Error> {
            // SAFETY: library is boxed and not moved here, therefore we can safley construct self-referential
            // struct here
            let library = Box::new(library);
            let init_fn = lookup_symbol::<InitFn>(&library, "tango_init")?;
            let count_fn = lookup_symbol::<CountFn>(&library, "tango_count")?;
            let select_fn = lookup_symbol::<SelectFn>(&library, "tango_select")?;
            let get_test_name_fn = lookup_symbol::<GetTestNameFn>(&library, "tango_get_test_name")?;
            let run_fn = lookup_symbol::<RunFn>(&library, "tango_run")?;
            let estimate_iterations_fn =
                lookup_symbol::<EstimateIterationsFn>(&library, "tango_estimate_iterations")?;
            let prepare_state_fn =
                lookup_symbol::<PrepareStateFn>(&library, "tango_prepare_state")?;
            let free_fn = lookup_symbol::<FreeFn>(&library, "tango_free")?;
            Ok(Self {
                _library: library,
                init_fn,
                count_fn,
                select_fn,
                get_test_name_fn,
                run_fn,
                estimate_iterations_fn,
                prepare_state_fn,
                free_fn,
            })
        }
    }

    impl VTable for LibraryVTable {
        fn init(&self) {
            unsafe { (self.init_fn)() }
        }

        fn count(&self) -> c_ulong {
            unsafe { (self.count_fn)() }
        }

        fn select(&self, func_idx: c_ulong) {
            unsafe { (self.select_fn)(func_idx) }
        }

        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut c_ulong) {
            unsafe { (self.get_test_name_fn)(ptr, len) }
        }

        fn run(&self, iterations: c_ulong) -> u64 {
            unsafe { (self.run_fn)(iterations) }
        }

        fn estimate_iterations(&self, time_ms: c_uint) -> c_ulong {
            unsafe { (self.estimate_iterations_fn)(time_ms) }
        }

        fn prepare_state(&self, seed: c_ulong) -> bool {
            unsafe { (self.prepare_state_fn)(seed) }
        }
    }

    impl Drop for LibraryVTable {
        fn drop(&mut self) {
            unsafe { (self.free_fn)() }
        }
    }

    fn lookup_symbol<'l, T>(
        library: &'l Library,
        name: &'static str,
    ) -> Result<Symbol<'static, T>, Error> {
        unsafe {
            let symbol: Symbol<'l, T> = library
                .get(name.as_bytes())
                .map_err(Error::UnableToLoadSymbol)?;
            Ok(mem::transmute(symbol))
        }
    }
}
