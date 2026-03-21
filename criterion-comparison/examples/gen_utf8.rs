use rand::{rng, seq::SliceRandom};
use std::{env, process};

/// Helper script that generates UTF-8 string with equal numbers of characters encoded
/// as 1, 2, 3, and 4 bytes
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <chars_per_class>", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().expect("first argument must be a number");

    assert!(one_byte_chars().all(|c| c.len_utf8() == 1));
    assert!(two_byte_chars().all(|c| c.len_utf8() == 2));
    assert!(three_byte_chars().all(|c| c.len_utf8() == 3));
    assert!(four_byte_chars().all(|c| c.len_utf8() == 4));

    let mut chars = one_byte_chars()
        .cycle()
        .take(n)
        .chain(two_byte_chars().cycle().take(n))
        .chain(three_byte_chars().cycle().take(n))
        .chain(four_byte_chars().cycle().take(n))
        .collect::<Vec<_>>();

    let mut rand = rng();
    chars.shuffle(&mut rand);

    let str = chars.iter().collect::<String>();
    println!("{str}");
}

// Character pools for each UTF-8 byte length
// see Unicode blocks: https://en.wikipedia.org/wiki/Unicode_block

fn one_byte_chars() -> impl Iterator<Item = char> + Clone {
    'A'..='z' // U+0000..U+007F – Basic Latin block
}

fn two_byte_chars() -> impl Iterator<Item = char> + Clone {
    '\u{0400}'..='\u{04FF}' // Cyrillic
}

fn three_byte_chars() -> impl Iterator<Item = char> + Clone {
    '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
}

fn four_byte_chars() -> impl Iterator<Item = char> + Clone {
    '\u{1F600}'..='\u{1F64F}' // Emoticons
}
