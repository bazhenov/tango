//! Loading and resolving symbols from .dylib/.so libraries

use self::ffi::{SelfVTable, VTable, SELF_SPI};
use crate::{Error, MeasureTarget};
use anyhow::Context;
use libloading::{Library, Symbol};
use std::{
    ffi::c_char,
    path::{Path, PathBuf},
    ptr::{addr_of, null},
    slice, str,
    sync::mpsc::{channel, Receiver, Sender},
    thread::{self, JoinHandle},
};

pub struct Spi<'l> {
    tests: Vec<NamedFunction>,
    vt: Box<dyn VTable + 'l>,
}

pub type FunctionIdx = usize;

#[derive(Debug, Clone)]
pub struct NamedFunction {
    pub name: String,

    /// Function index in FFI API
    pub idx: FunctionIdx,
}

pub(crate) struct SpiHandle {
    worker: JoinHandle<()>,
    tests: Vec<NamedFunction>,
    selected_function: Option<FunctionIdx>,
    tx: Sender<SpiRequest>,
    rx: Receiver<SpiReply>,
}

impl SpiHandle {
    pub(crate) fn tests(&self) -> &[NamedFunction] {
        &self.tests
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<&NamedFunction> {
        self.tests.iter().find(|f| f.name == name)
    }

    pub(crate) fn run(&mut self, func: FunctionIdx, iterations: usize) -> u64 {
        self.select(func);
        self.tx.send(SpiRequest::Run { iterations }).unwrap();
        match self.rx.recv().unwrap() {
            SpiReply::Run(time) => time,
            r @ _ => panic!("Unexpected response: {:?}", r),
        }
    }

    pub(crate) fn measure(&mut self, func: FunctionIdx, iterations: usize) {
        self.select(func);
        self.tx.send(SpiRequest::Measure { iterations }).unwrap();
    }

    pub(crate) fn read_sample(&mut self) -> u64 {
        match self.rx.recv().unwrap() {
            SpiReply::Measure(time) => time,
            r @ _ => panic!("Unexpected response: {:?}", r),
        }
    }

    pub(crate) fn estimate_iterations(&mut self, func: FunctionIdx, time_ms: u32) -> usize {
        self.select(func);
        self.tx
            .send(SpiRequest::EstimateIterations { time_ms })
            .unwrap();
        match self.rx.recv().unwrap() {
            SpiReply::EstimateIterations(iters) => iters,
            r @ _ => panic!("Unexpected response: {:?}", r),
        }
    }

    pub(crate) fn prepare_state(&mut self, func: FunctionIdx, seed: u64) -> bool {
        self.select(func);
        self.tx.send(SpiRequest::PrepareState { seed }).unwrap();
        match self.rx.recv().unwrap() {
            SpiReply::PrepareState(ret) => ret,
            r @ _ => panic!("Unexpected response: {:?}", r),
        }
    }

    pub(crate) fn select(&mut self, idx: usize) {
        let different_function = self.selected_function.map(|v| v != idx).unwrap_or(true);
        if different_function {
            self.tx.send(SpiRequest::Select { idx }).unwrap();
            match self.rx.recv().unwrap() {
                SpiReply::Select(_) => {
                    self.selected_function = Some(idx);
                    return;
                }
                r @ _ => panic!("Unexpected response: {:?}", r),
            }
        }
    }
}

fn spi_worker_for_library(path: PathBuf, rx: Receiver<SpiRequest>, tx: Sender<SpiReply>) {
    let lib = unsafe { Library::new(&path) }
        .with_context(|| format!("Unable to open library: {}", path.display()))
        .unwrap();
    let spi = Spi::for_vtable(ffi::LibraryVTable::new(&lib).unwrap()).unwrap();
    spi_worker(rx, spi, tx);
}

fn spi_worker_for_self(self_vtable: SelfVTable, rx: Receiver<SpiRequest>, tx: Sender<SpiReply>) {
    let spi = Spi::for_vtable(self_vtable).unwrap();
    spi_worker(rx, spi, tx);
}

fn spi_worker(rx: Receiver<SpiRequest>, spi: Spi<'_>, tx: Sender<SpiReply>) {
    use SpiReply as Rp;
    use SpiRequest as Rq;

    while let Ok(req) = rx.recv() {
        let reply = match req {
            Rq::EstimateIterations { time_ms } => {
                Rp::EstimateIterations(spi.vt.estimate_iterations(time_ms))
            }
            Rq::PrepareState { seed } => Rp::PrepareState(spi.vt.prepare_state(seed)),
            Rq::Select { idx } => Rp::Select(spi.vt.select(idx)),
            Rq::Run { iterations } => Rp::Run(spi.vt.run(iterations)),
            Rq::Measure { iterations } => Rp::Measure(spi.vt.run(iterations)),
            Rq::ListTests => Rp::ListTests(spi.tests.to_vec()),
        };
        tx.send(reply).unwrap();
    }
}

impl<'l> Spi<'l> {
    pub(crate) fn spi_handle_for_library(path: impl AsRef<Path>) -> SpiHandle {
        let (request_tx, request_rx) = channel();
        let (reply_tx, reply_rx) = channel();
        let path = path.as_ref().to_path_buf();
        let worker = thread::spawn(move || spi_worker_for_library(path, request_rx, reply_tx));

        request_tx.send(SpiRequest::ListTests).unwrap();
        let tests = match reply_rx.recv().unwrap() {
            SpiReply::ListTests(tests) => tests,
            r @ _ => panic!("Unexpected response: {:?}", r),
        };

        SpiHandle {
            worker,
            tests,
            tx: request_tx,
            rx: reply_rx,
            selected_function: None,
        }
    }

    pub(crate) fn spi_handle_for_self() -> Option<SpiHandle> {
        let (request_tx, request_rx) = channel();
        let (reply_tx, reply_rx) = channel();
        let vtable = unsafe { SELF_SPI.take() }?;
        let worker = thread::spawn(move || spi_worker_for_self(vtable, request_rx, reply_tx));

        request_tx.send(SpiRequest::ListTests).unwrap();
        let tests = match reply_rx.recv().unwrap() {
            SpiReply::ListTests(tests) => tests,
            r @ _ => panic!("Unexpected response: {:?}", r),
        };

        Some(SpiHandle {
            worker,
            tests,
            tx: request_tx,
            rx: reply_rx,
            selected_function: None,
        })
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
}

enum SpiRequest {
    ListTests,
    EstimateIterations { time_ms: u32 },
    PrepareState { seed: u64 },
    Select { idx: usize },
    Run { iterations: usize },
    Measure { iterations: usize },
}

#[derive(Debug)]
enum SpiReply {
    ListTests(Vec<NamedFunction>),
    EstimateIterations(usize),
    PrepareState(bool),
    Select(()),
    Run(u64),
    Measure(u64),
}

/// State which holds the information about list of benchmarks and which one is selected.
/// Used in FFI API (`tango_*` functions).
pub struct State {
    pub benchmarks: Vec<Box<dyn MeasureTarget>>,
    pub selected_function: usize,
}

impl State {
    fn selected(&self) -> &dyn MeasureTarget {
        self.benchmarks[self.selected_function].as_ref()
    }

    fn selected_mut(&mut self) -> &mut dyn MeasureTarget {
        self.benchmarks[self.selected_function].as_mut()
    }
}

/// Global state of the benchmarking library
static mut STATE: Option<State> = None;

/// `tango_init()` implementation
///
/// This function is not exported from the library, but is used by the `tango_init()` functions
/// generated by the `tango_benchmark!()` macro.
pub unsafe fn __tango_init(benchmarks: Vec<Box<dyn MeasureTarget>>) {
    if STATE.is_none() {
        STATE = Some(State {
            benchmarks,
            selected_function: 0,
        });
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
    use std::{os::raw::c_char, ptr::null, usize};

    /// Signature types of all FFI API functions
    pub type InitFn = unsafe extern "C" fn();
    type CountFn = unsafe extern "C" fn() -> usize;
    type GetTestNameFn = unsafe extern "C" fn(*mut *const c_char, *mut usize);
    type SelectFn = unsafe extern "C" fn(usize);
    type RunFn = unsafe extern "C" fn(usize) -> u64;
    type EstimateIterationsFn = unsafe extern "C" fn(u32) -> usize;
    type PrepareStateFn = unsafe extern "C" fn(u64) -> bool;
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
    unsafe extern "C" fn tango_prepare_state(seed: u64) -> bool {
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
        fn count(&self) -> usize;
        fn select(&self, func_idx: usize);
        fn get_test_name(&self, ptr: *mut *const c_char, len: *mut usize);
        fn run(&self, iterations: usize) -> u64;
        fn estimate_iterations(&self, time_ms: u32) -> usize;
        fn prepare_state(&self, seed: u64) -> bool;
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
            // In executable mode `tango_init` is already called by the main function
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

    pub(super) struct LibraryVTable<'l> {
        init_fn: Symbol<'l, InitFn>,
        count_fn: Symbol<'l, CountFn>,
        select_fn: Symbol<'l, SelectFn>,
        get_test_name_fn: Symbol<'l, GetTestNameFn>,
        run_fn: Symbol<'l, RunFn>,
        estimate_iterations_fn: Symbol<'l, EstimateIterationsFn>,
        prepare_state_fn: Symbol<'l, PrepareStateFn>,
        free_fn: Symbol<'l, FreeFn>,
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
                    prepare_state_fn: lookup_symbol(library, "tango_prepare_state")?,
                    free_fn: lookup_symbol(library, "tango_free")?,
                })
            }
        }
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

        fn prepare_state(&self, seed: u64) -> bool {
            unsafe { (self.prepare_state_fn)(seed) }
        }
    }

    impl<'l> Drop for LibraryVTable<'l> {
        fn drop(&mut self) {
            unsafe { (self.free_fn)() }
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
