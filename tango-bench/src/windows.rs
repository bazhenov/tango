//! Windows-specific functionality for loading EXE files as libraries
//!
//! When loading an executable with `LoadLibrary()` on Windows, the Import Address Table (IAT)
//! is not properly configured. This module provides functionality to patch the IAT so that
//! all imported function calls work correctly.
//!
//! The IAT patching process:
//! 1. Parse the PE headers to find the import directory
//! 2. For each imported DLL, load it with `LoadLibrary`
//! 3. For each imported function, resolve its address with `GetProcAddress`
//! 4. Write the resolved addresses to the IAT entries

use std::{
    ffi::{c_void, CStr},
    ptr,
};
use thiserror::Error;
use windows::{
    core::PCSTR,
    Win32::{
        Foundation::HMODULE,
        System::{
            Diagnostics::Debug::{IMAGE_DIRECTORY_ENTRY_IMPORT, IMAGE_NT_HEADERS64},
            LibraryLoader::{GetProcAddress, LoadLibraryA},
            Memory::{VirtualProtect, PAGE_PROTECTION_FLAGS, PAGE_READWRITE},
            SystemServices::{
                IMAGE_DOS_HEADER, IMAGE_DOS_SIGNATURE, IMAGE_IMPORT_DESCRIPTOR, IMAGE_NT_SIGNATURE,
                IMAGE_ORDINAL_FLAG64,
            },
        },
    },
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("Invalid DOS signature")]
    InvalidDosSignature,

    #[error("Invalid NT signature")]
    InvalidNtSignature,

    #[error("No import directory found")]
    NoImportDirectory,

    #[error("Failed to load dependency: {0}")]
    FailedToLoadDependency(String),

    #[error("Failed to resolve function: {0}")]
    FailedToResolveFunction(String),

    #[error("Failed to change memory protection")]
    FailedToChangeProtection,
}

/// Patches the Import Address Table (IAT) of a loaded module
///
/// This function should be called immediately after loading an EXE file with `LoadLibrary`.
/// It resolves all imported functions and writes their addresses to the IAT.
///
/// # Safety
///
/// This function is unsafe because it:
/// - Reads and writes raw memory based on PE header offsets
/// - Changes memory protection of IAT pages
/// - Assumes the module handle points to a valid PE image
pub unsafe fn patch_iat(module: HMODULE) -> Result<(), Error> {
    let base = module.0;

    // Parse DOS header
    let dos_header = &*(base as *const IMAGE_DOS_HEADER);
    if dos_header.e_magic != IMAGE_DOS_SIGNATURE {
        return Err(Error::InvalidDosSignature);
    }

    // Parse NT headers
    let nt_headers = &*(base.offset(dos_header.e_lfanew as isize) as *const IMAGE_NT_HEADERS64);
    if nt_headers.Signature != IMAGE_NT_SIGNATURE {
        return Err(Error::InvalidNtSignature);
    }

    // Get import directory
    let import_dir =
        &nt_headers.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT.0 as usize];
    if import_dir.VirtualAddress == 0 {
        // No imports - nothing to patch
        return Ok(());
    }

    // Iterate through import descriptors
    let mut import_desc =
        base.offset(import_dir.VirtualAddress as isize) as *const IMAGE_IMPORT_DESCRIPTOR;

    while (*import_desc).Name != 0 {
        // Get the DLL name
        let dll_name_ptr = base.offset((*import_desc).Name as isize) as *const i8;

        // Load the dependency DLL
        let Ok(dll_handle) = LoadLibraryA(PCSTR(dll_name_ptr as *const u8)) else {
            let dll_name = CStr::from_ptr(dll_name_ptr);
            return Err(Error::FailedToLoadDependency(
                dll_name.to_string_lossy().into_owned(),
            ));
        };

        // Get the Import Name Table (INT) and Import Address Table (IAT)
        // OriginalFirstThunk points to INT (names), FirstThunk points to IAT (addresses to patch)
        let mut int_entry =
            base.offset((*import_desc).Anonymous.OriginalFirstThunk as isize) as *const u64;
        let mut iat_entry = base.offset((*import_desc).FirstThunk as isize) as *mut u64;

        while *int_entry != 0 {
            let func_addr = if (*int_entry & IMAGE_ORDINAL_FLAG64) != 0 {
                // Import by ordinal
                let ordinal = (*int_entry & 0xFFFF) as u16;
                let Some(func_addr) =
                    GetProcAddress(dll_handle, PCSTR(ordinal as usize as *const u8))
                else {
                    let name = format!("ordinal {}", *int_entry & 0xFFFF);
                    return Err(Error::FailedToResolveFunction(name));
                };
                func_addr
            } else {
                // Import by name
                // The INT entry points to IMAGE_IMPORT_BY_NAME structure
                // First 2 bytes are hint, followed by null-terminated name
                let import_by_name = base.offset(*int_entry as isize);
                let func_name_ptr = import_by_name.offset(2) as *const i8;
                let Some(func_addr) = GetProcAddress(dll_handle, PCSTR(func_name_ptr as *const u8))
                else {
                    let name = CStr::from_ptr(func_name_ptr).to_string_lossy().into_owned();
                    return Err(Error::FailedToResolveFunction(name));
                };
                func_addr
            };

            // Change memory protection to allow writing
            let mut old_protect = PAGE_PROTECTION_FLAGS::default();
            if VirtualProtect(
                iat_entry as *const c_void,
                std::mem::size_of::<u64>(),
                PAGE_READWRITE,
                &mut old_protect,
            )
            .is_err()
            {
                return Err(Error::FailedToChangeProtection);
            }

            // Write the resolved address
            ptr::write_volatile(iat_entry, func_addr as usize as u64);

            // Restore original protection
            let mut dummy = PAGE_PROTECTION_FLAGS::default();
            let _ = VirtualProtect(
                iat_entry as *const c_void,
                std::mem::size_of::<u64>(),
                old_protect,
                &mut dummy,
            );

            int_entry = int_entry.offset(1);
            iat_entry = iat_entry.offset(1);
        }

        import_desc = import_desc.offset(1);
    }

    Ok(())
}
