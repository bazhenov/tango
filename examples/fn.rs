use std::{mem, ptr};

use mmap::{MapOption, MemoryMap};

fn main() {
    let f: fn() = foo;

    let buffer_from: *const u8 = unsafe { mem::transmute::<_, _>(f) };
    let mmap = unsafe { reflect(buffer_from, 50) };
    let fn_copy: fn() = unsafe { mem::transmute(mmap.data()) };

    println!("Function pointer: {:p}", f);
    println!("Function-copy pointer: {:p}", fn_copy);

    println!("Function data: {}", unsafe { mem::transmute::<_, &u8>(f) });
    println!("Function copy data: {}", unsafe {
        mem::transmute::<_, &u8>(fn_copy)
    });

    f();

    // println!("Buffer pointer: {:p}", buffer_to.as_ptr());
    fn_copy();
}

fn foo() {
    println!("Hello world");
}

unsafe fn reflect(instructions: *const u8, size: usize) -> MemoryMap {
    let map = MemoryMap::new(
        size,
        &[
            // MapOption::MapAddr(0 as *mut u8),
            // MapOption::MapOffset(0),
            MapOption::MapFd(-1),
            MapOption::MapReadable,
            MapOption::MapWritable,
            MapOption::MapExecutable,
            // MapOption::MapNonStandardFlags(libc::MAP_ANON),
            // MapOption::MapNonStandardFlags(libc::MAP_PRIVATE),
        ],
    )
    .unwrap();

    ptr::copy_nonoverlapping(instructions, map.data(), size);

    map
}
