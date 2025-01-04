//! Loading and resolving symbols from .dylib/.so libraries

use self::ffi::SELF_VTABLE;
use crate::{Benchmark, ErasedSampler, Error};
use anyhow::Context;
use ffi::VTable;
use libloading::{Library, Symbol};
use std::{
    cell::UnsafeCell,
    ffi::{c_char, c_ulonglong},
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
    // Benchmarks are executed synchronously when calling SPI
    //
    // Dispatcher switches between baseline and candidate after each sample
    Synchronous,

    // Benchmarks are executed in different threads
    //
    // Dispatcher creates a separate thread for baseline and candidate, but synchronize them after each benchmark
    Asynchronous,
}

enum SpiMode {
    Synchronous {
        vt: VTable,
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
        spi_handle_for_vtable(ffi::VTable::new(lib).unwrap(), mode)
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

    pub(crate) fn run(&mut self, iterations: usize) -> u64 {
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.run(iterations as c_ulonglong),
            SpiMode::Asynchronous { worker: _, tx, rx } => {
                tx.send(SpiRequest::Run { iterations }).unwrap();
                match rx.recv().unwrap() {
                    SpiReply::Run(time) => time,
                    r => panic!("Unexpected response: {:?}", r),
                }
            }
        }
    }

    pub(crate) fn measure(&mut self, iterations: usize) {
        match &mut self.mode {
            SpiMode::Synchronous {
                vt,
                last_measurement,
            } => {
                *last_measurement = vt.run(iterations as c_ulonglong);
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

    pub(crate) fn estimate_iterations(&mut self, time_ms: u32) -> Result<usize, Error> {
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.estimate_iterations(time_ms),
            SpiMode::Asynchronous { tx, rx, .. } => {
                tx.send(SpiRequest::EstimateIterations { time_ms }).unwrap();
                match rx.recv().unwrap() {
                    SpiReply::EstimateIterations(iters) => iters,
                    r => panic!("Unexpected response: {:?}", r),
                }
            }
        }
    }

    pub(crate) fn prepare_state(&mut self, seed: u64) -> Result<(), Error> {
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.prepare_state(seed),
            SpiMode::Asynchronous { tx, rx, .. } => {
                tx.send(SpiRequest::PrepareState { seed }).unwrap();
                match rx.recv().unwrap() {
                    SpiReply::PrepareState(result) => result,
                    r => panic!("Unexpected response: {:?}", r),
                }
            }
        }
    }

    pub(crate) fn select(&mut self, idx: usize) {
        match &self.mode {
            SpiMode::Synchronous { vt, .. } => vt.select(idx as c_ulonglong),
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

fn spi_worker(vt: &VTable, rx: Receiver<SpiRequest>, tx: Sender<SpiReply>) {
    use SpiReply as Rp;
    use SpiRequest as Rq;

    while let Ok(req) = rx.recv() {
        let reply = match req {
            Rq::EstimateIterations { time_ms } => {
                Rp::EstimateIterations(vt.estimate_iterations(time_ms))
            }
            Rq::PrepareState { seed } => Rp::PrepareState(vt.prepare_state(seed)),
            Rq::Select { idx } => {
                vt.select(idx as c_ulonglong);
                Rp::Select
            }
            Rq::Run { iterations } => Rp::Run(vt.run(iterations as c_ulonglong)),
            Rq::Measure { iterations } => Rp::Measure(vt.run(iterations as c_ulonglong)),
            Rq::Shutdown => break,
        };
        tx.send(reply).unwrap();
    }
}

fn spi_handle_for_vtable(vt: VTable, mode: SpiModeKind) -> Spi {
    vt.init();
    let tests = enumerate_tests(&vt).unwrap();

    match mode {
        SpiModeKind::Asynchronous => {
            let (request_tx, request_rx) = channel();
            let (reply_tx, reply_rx) = channel();
            let worker = thread::spawn(move || {
                spi_worker(&vt, request_rx, reply_tx);
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
                vt,
                last_measurement: 0,
            },
        },
    }
}

fn enumerate_tests(vt: &VTable) -> Result<Vec<NamedFunction>, Error> {
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
    EstimateIterations(Result<usize, Error>),
    PrepareState(Result<(), Error>),
    Select,
    Run(u64),
    Measure(u64),
}

/// State which holds the information about list of benchmarks and which one is selected.
/// Used in FFI API (`tango_*` functions).
struct State {
    benchmarks: Vec<Benchmark>,
    selected_function: Option<(usize, Option<Box<dyn ErasedSampler>>)>,
    last_error: Option<String>,
}

impl State {
    fn selected(&self) -> &Benchmark {
        &self.benchmarks[self.ensure_selected()]
    }

    fn ensure_selected(&self) -> usize {
        self.selected_function
            .as_ref()
            .map(|(idx, _)| *idx)
            .expect("No function was selected. Call tango_select() first")
    }

    fn selected_state_mut(&mut self) -> Option<&mut Box<dyn ErasedSampler>> {
        self.selected_function
            .as_mut()
            .and_then(|(_, state)| state.as_mut())
    }
}

/// Global state of the benchmarking library
static STATE: StateWrapper = StateWrapper(UnsafeCell::new(None));

struct StateWrapper(UnsafeCell<Option<State>>);
unsafe impl Sync for StateWrapper {}

impl StateWrapper {
    unsafe fn as_ref(&self) -> Option<&State> {
        (*self.0.get()).as_ref()
    }

    unsafe fn as_mut(&self) -> Option<&mut State> {
        (*self.0.get()).as_mut()
    }
}

/// `tango_init()` implementation
///
/// This function is not exported from the library, but is used by the `tango_init()` functions
/// generated by the `tango_benchmark!()` macro.
pub fn __tango_init(benchmarks: Vec<Benchmark>) {
    if unsafe { STATE.as_ref().is_none() } {
        let state = Some(State {
            benchmarks,
            selected_function: None,
            last_error: None,
        });
        unsafe { *STATE.0.get() = state }
    }
}

/// Defines all the foundation types and exported symbols for the FFI communication API between two
/// executables.
///
/// Tango execution model implies simultaneous execution of the code from two binaries. To achieve that
/// Tango benchmark is compiled in a way that executable is also a shared library (.dll, .so, .dylib). This
/// way two executables can coexist in the single process at the same time.
pub mod ffi {
    use super::*;
    use std::{
        ffi::{c_int, c_uint, c_ulonglong},
        os::raw::c_char,
        panic::{catch_unwind, UnwindSafe},
        ptr::null,
    };

    /// Signature types of all FFI API functions
    pub type InitFn = unsafe extern "C" fn();
    type CountFn = unsafe extern "C" fn() -> c_ulonglong;
    type GetTestNameFn = unsafe extern "C" fn(*mut *const c_char, *mut c_ulonglong);
    type SelectFn = unsafe extern "C" fn(c_ulonglong);
    type RunFn = unsafe extern "C" fn(c_ulonglong) -> u64;
    type EstimateIterationsFn = unsafe extern "C" fn(c_uint) -> c_ulonglong;
    type PrepareStateFn = unsafe extern "C" fn(c_ulonglong) -> c_int;
    type GetLastErrorFn = unsafe extern "C" fn(*mut *const c_char, *mut c_ulonglong) -> c_int;
    type FreeFn = unsafe extern "C" fn();

    #[no_mangle]
    unsafe extern "C" fn tango_count() -> c_ulonglong {
        STATE
            .as_ref()
            .map(|s| s.benchmarks.len() as c_ulonglong)
            .unwrap_or(0)
    }

    #[no_mangle]
    unsafe extern "C" fn tango_select(idx: c_ulonglong) {
        if let Some(s) = STATE.as_mut() {
            let idx = idx as usize;
            assert!(idx < s.benchmarks.len());

            s.selected_function = Some(match s.selected_function.take() {
                // Preserving state if the same function is selected
                Some((selected, state)) if selected == idx => (selected, state),
                _ => (idx, None),
            });
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_get_test_name(name: *mut *const c_char, length: *mut c_ulonglong) {
        if let Some(s) = STATE.as_ref() {
            let n = s.selected().name();
            *name = n.as_ptr() as _;
            *length = n.len() as _;
        } else {
            *name = null();
            *length = 0;
        }
    }

    /// Returns C-string to a description of last error (if any)
    ///
    /// Returns: 0 if last error was returned, -1 otherwise
    #[no_mangle]
    unsafe extern "C" fn tango_get_last_error(
        name: *mut *const c_char,
        length: *mut c_ulonglong,
    ) -> c_int {
        if let Some(err) = STATE.as_ref().and_then(|s| s.last_error.as_ref()) {
            *name = err.as_ptr() as _;
            *length = err.len() as _;
            0
        } else {
            *name = null();
            *length = 0;
            -1
        }
    }

    #[no_mangle]
    unsafe extern "C" fn tango_run(iterations: c_ulonglong) -> u64 {
        catch(|| {
            if let Some(s) = STATE.as_mut() {
                s.selected_state_mut()
                    .expect("no tango_prepare_state() was called")
                    .measure(iterations as usize)
            } else {
                0
            }
        })
        .unwrap_or(0)
    }

    /// Returns an estimation of number of iterations needed to spent given amount of time
    ///
    /// Returns: the number of iterations (minimum of 1) or 0 if error happens during building the estimate.
    #[no_mangle]
    unsafe extern "C" fn tango_estimate_iterations(time_ms: c_uint) -> c_ulonglong {
        catch(|| {
            if let Some(s) = STATE.as_mut() {
                s.selected_state_mut()
                    .expect("no tango_prepare_state() was called")
                    .as_mut()
                    .estimate_iterations(time_ms)
                    .max(1) as c_ulonglong
            } else {
                0
            }
        })
        .unwrap_or(0)
    }

    /// Prepares benchmark internal state
    ///
    /// Should be called once benchmark was selected ([`tango_select`]) to initialize all needed state.
    ///
    /// Returns: 0 if success, otherwise preparing state was failed
    #[no_mangle]
    unsafe extern "C" fn tango_prepare_state(seed: c_ulonglong) -> c_int {
        catch(|| {
            if let Some(s) = STATE.as_mut() {
                let Some((idx, state)) = &mut s.selected_function else {
                    panic!("No tango_select() was called")
                };
                *state = Some(s.benchmarks[*idx].prepare_state(seed));
            }
            0
        })
        .unwrap_or(-1)
    }

    #[no_mangle]
    unsafe extern "C" fn tango_free() {
        unsafe { *STATE.0.get() = None }
    }

    /// Since unwinding cannot cross FFI boundaries, we catch all panics here
    /// and print their message for debugging, while returning None to indicate failure.
    fn catch<T>(f: impl FnOnce() -> T + UnwindSafe) -> Option<T> {
        match catch_unwind(f) {
            Ok(r) => Some(r),
            Err(e) => {
                // Here we're assuming state is already initialized, because f was running some operations on it
                let state = unsafe { STATE.as_mut().unwrap() };
                if let Some(msg) = e.downcast_ref::<&str>() {
                    state.last_error = Some(msg.to_string());
                } else {
                    state.last_error = e.downcast_ref::<String>().cloned();
                }
                None
            }
        }
    }

    pub(super) static mut SELF_VTABLE: Option<VTable> = Some(VTable::for_self());

    pub(super) struct VTable {
        /// SAFETY: using plain function pointers instead of [`Symbol`] here to generalize over case
        /// when we have to have VTable for function defined in our own address space (so called self VTable) –
        /// [`Self::for_self()`].
        ///
        /// This is is sound because:
        ///  (1) this struct is private and field can not be accessed outside
        ///  (2) rust has drop order guarantee (fields are dropped in declaration order)
        init_fn: InitFn,
        count_fn: CountFn,
        select_fn: SelectFn,
        get_test_name_fn: GetTestNameFn,
        get_last_error_fn: GetLastErrorFn,
        run_fn: RunFn,
        estimate_iterations_fn: EstimateIterationsFn,
        prepare_state_fn: PrepareStateFn,
        free_fn: FreeFn,

        /// SAFETY: This field should be last because it should be dropped last
        _library: Option<Box<Library>>,
    }

    impl VTable {
        pub(super) fn new(library: Library) -> Result<Self, Error> {
            // SAFETY: library is boxed and not moved here, therefore we can safley construct self-referential
            // struct here
            let library = Box::new(library);
            let init_fn = *lookup_symbol::<InitFn>(&library, "tango_init")?;
            let count_fn = *lookup_symbol::<CountFn>(&library, "tango_count")?;
            let select_fn = *lookup_symbol::<SelectFn>(&library, "tango_select")?;
            let get_test_name_fn =
                *lookup_symbol::<GetTestNameFn>(&library, "tango_get_test_name")?;
            let run_fn = *lookup_symbol::<RunFn>(&library, "tango_run")?;
            let estimate_iterations_fn =
                *lookup_symbol::<EstimateIterationsFn>(&library, "tango_estimate_iterations")?;
            let prepare_state_fn =
                *lookup_symbol::<PrepareStateFn>(&library, "tango_prepare_state")?;
            let get_last_error_fn =
                *lookup_symbol::<GetLastErrorFn>(&library, "tango_get_last_error")?;
            let free_fn = *lookup_symbol::<FreeFn>(&library, "tango_free")?;
            Ok(Self {
                init_fn,
                count_fn,
                select_fn,
                get_test_name_fn,
                run_fn,
                estimate_iterations_fn,
                prepare_state_fn,
                get_last_error_fn,
                free_fn,
                _library: Some(library),
            })
        }

        const fn for_self() -> Self {
            unsafe extern "C" fn no_init() {
                // In executable mode `tango_init` is already called by the main function
            }
            Self {
                init_fn: no_init,
                count_fn: ffi::tango_count,
                select_fn: ffi::tango_select,
                get_test_name_fn: ffi::tango_get_test_name,
                run_fn: ffi::tango_run,
                estimate_iterations_fn: ffi::tango_estimate_iterations,
                prepare_state_fn: ffi::tango_prepare_state,
                get_last_error_fn: ffi::tango_get_last_error,
                free_fn: ffi::tango_free,
                _library: None,
            }
        }

        pub(super) fn init(&self) {
            unsafe { (self.init_fn)() }
        }

        pub(super) fn count(&self) -> c_ulonglong {
            unsafe { (self.count_fn)() }
        }

        pub(super) fn select(&self, func_idx: c_ulonglong) {
            unsafe { (self.select_fn)(func_idx) }
        }

        pub(super) fn get_test_name(&self, ptr: *mut *const c_char, len: *mut c_ulonglong) {
            unsafe { (self.get_test_name_fn)(ptr, len) }
        }

        pub(super) fn run(&self, iterations: c_ulonglong) -> u64 {
            unsafe { (self.run_fn)(iterations) }
        }

        pub(super) fn estimate_iterations(&self, time_ms: c_uint) -> Result<usize, Error> {
            match unsafe { (self.estimate_iterations_fn)(time_ms) } {
                0 => Err(self.last_error()?),
                iters => Ok(iters as usize),
            }
        }

        pub(super) fn prepare_state(&self, seed: c_ulonglong) -> Result<(), Error> {
            match unsafe { (self.prepare_state_fn)(seed) } {
                0 => Ok(()),
                _ => Err(self.last_error()?),
            }
        }

        fn last_error(&self) -> Result<Error, Error> {
            let mut length = 0;
            let mut name = null();
            if unsafe { (self.get_last_error_fn)(&mut name, &mut length) } != 0 {
                Err(Error::UnknownFFIError)
            } else {
                let name = unsafe { slice::from_raw_parts(name as *const u8, length as usize) };
                str::from_utf8(name)
                    .map_err(Error::InvalidFFIString)
                    .map(str::to_string)
                    .map(Error::FFIError)
            }
        }
    }

    impl Drop for VTable {
        fn drop(&mut self) {
            unsafe { (self.free_fn)() }
        }
    }

    fn lookup_symbol<'l, T>(
        library: &'l Library,
        name: &'static str,
    ) -> Result<Symbol<'l, T>, Error> {
        unsafe {
            library
                .get(name.as_bytes())
                .map_err(Error::UnableToLoadSymbol)
        }
    }
}
