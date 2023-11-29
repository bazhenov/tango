use std::path::{Path, PathBuf};

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
/// of pairwise benchmarking.
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
) -> Option<PathBuf> {
    #[cfg(not(target_os = "linux"))]
    {
        None
    }

    #[cfg(target_os = "linux")]
    {
        use goblin::{
            elf::{Dyn, Elf},
            elf64::{
                dynamic::{DF_1_PIE, DT_FLAGS_1},
                program_header::PT_DYNAMIC,
            },
        };
        use scroll::{Pread, Pwrite};
        use std::{fs, mem};
        let mut bytes = fs::read(path.as_ref()).unwrap();
        let elf = Elf::parse(&bytes).unwrap();

        let Some(dynamic) = elf.dynamic else {
            return None;
        };
        if dynamic.info.flags_1 & DF_1_PIE == 0 {
            return None;
        }

        let (dyn_idx, _) = dynamic
            .dyns
            .iter()
            .enumerate()
            .find(|(_, d)| d.d_tag == DT_FLAGS_1)
            .expect("Unable to found DT_FLAGS_1 position");

        // Finding PT_DYNAMIC section offset
        let header = elf
            .program_headers
            .iter()
            .find(|h| h.p_type == PT_DYNAMIC)
            .expect("Unable to find PT_DYNAMIC section");
        // Finding target Dyn item offset
        let dyn_offset = header.p_offset as usize + dyn_idx * mem::size_of::<Dyn>();

        // Crosschecking we found right dyn tag
        let mut dyn_item = bytes.pread::<Dyn>(dyn_offset).unwrap();
        assert!(
            dyn_item.d_tag == DT_FLAGS_1,
            "DT_FLAGS_1 flag crosscheck failed"
        );
        assert!(
            dyn_item.d_val == dynamic.info.flags_1,
            "DT_FLAGS_1 flag crosscheck failed"
        );

        let path = path.as_ref().with_extension("patched");

        // clearing DF_1_PIE bit and writing patched binary
        dyn_item.d_val &= !DF_1_PIE;
        bytes.pwrite(dyn_item, dyn_offset).unwrap();
        fs::write(&path, bytes).unwrap();

        Some(path)
    }
}