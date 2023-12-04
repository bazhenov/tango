use goblin::{
    elf::{Dyn, Elf},
    elf64::{
        dynamic::{DF_1_PIE, DT_FLAGS_1},
        program_header::PT_DYNAMIC,
    },
    error::Error as GoblinError,
};
use scroll::{Pread, Pwrite};
use std::{
    fs, mem,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unable to parse ELF file")]
    UnableToParseElf(GoblinError),

    #[error("Unable to serialize ELF file")]
    UnableToSerializeElf(GoblinError),

    #[error("Unable to found DT_FLAGS_1 position")]
    NoDTFlags1Found,

    #[error("Unable to find PT_DYNAMIC section")]
    NoDynamicSectionFound,

    #[error("DT_FLAGS_1 flag crosscheck failed")]
    FlagCrosscheckFailed,

    #[error("IOError")]
    IOError(#[from] std::io::Error),
}

/// Patches executable file for new version of glibc dynamic loader
///
/// After glibc 2.29 on linux `dlopen` is explicitly denying loading
/// PIE executables as a shared library. The following error might appear:
///
/// ```console
/// dlopen error: cannot dynamically load position-independent executable
/// ```
///
/// From 2.29 [dynamic loader throws an error](glibc) if `DF_1_PIE` flag is set in
/// `DT_FLAG_1` tag on the `PT_DYNAMIC` section. Although the loading of executable
/// files as a shared library was never an intended use case, through the years
/// some applications adopted this technique and it is very convinient in the context
/// of paired benchmarking.
///
/// Following method check if this flag is set and patch binary at runtime
/// (writing patched version in a different file). As far as I am aware
/// this is safe modification because `DF_1_PIE` is purely informational and doesn't
/// changle the dynamic linking process in any way. Theoretically in the future this modification
/// could prevent ASLR ramndomization on the OS level which is irrelevant for benchmark
/// executables.
///
/// [glibc]: https://github.com/bminor/glibc/blob/2e0c0ff95ca0e3122eb5b906ee26a31f284ce5ab/elf/dl-load.c#L1280-L1282
pub fn patch_pie_binary_if_needed(
    #[allow(unused_variables)] path: impl AsRef<Path>,
) -> Result<Option<PathBuf>, Error> {
    let mut bytes = fs::read(path.as_ref())?;
    let elf = Elf::parse(&bytes).map_err(Error::UnableToParseElf)?;

    let Some(dynamic) = elf.dynamic else {
        return Ok(None);
    };
    if dynamic.info.flags_1 & DF_1_PIE == 0 {
        return Ok(None);
    }

    let (dyn_idx, _) = dynamic
        .dyns
        .iter()
        .enumerate()
        .find(|(_, d)| d.d_tag == DT_FLAGS_1)
        .ok_or(Error::NoDTFlags1Found)?;

    // Finding PT_DYNAMIC section offset
    let header = elf
        .program_headers
        .iter()
        .find(|h| h.p_type == PT_DYNAMIC)
        .ok_or(Error::NoDynamicSectionFound)?;

    // Finding target Dyn item offset
    let dyn_offset = header.p_offset as usize + dyn_idx * mem::size_of::<Dyn>();

    // Crosschecking we found right dyn tag
    let mut dyn_item = bytes
        .pread::<Dyn>(dyn_offset)
        .map_err(Error::UnableToSerializeElf)?;

    if dyn_item.d_tag != DT_FLAGS_1 || dyn_item.d_val != dynamic.info.flags_1 {
        return Err(Error::FlagCrosscheckFailed);
    }

    // clearing DF_1_PIE bit and writing patched binary
    dyn_item.d_val &= !DF_1_PIE;
    bytes
        .pwrite(dyn_item, dyn_offset)
        .map_err(Error::UnableToSerializeElf)?;

    let path = path.as_ref().with_extension("patched");
    fs::write(&path, bytes)?;

    Ok(Some(path))
}
