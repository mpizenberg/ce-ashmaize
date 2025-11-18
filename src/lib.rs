/*!
# Ashmaize: a widely portable ASIC resistant hash algorithm

AshMaize is a simple PoW that is somewhat ASIC resistant, yet relative simple to implement

# How to use

## The [`Rom`]

First you need to initialise the [`Rom`]. It is the _Read Only Memory_
and it is generated once and can be reused for different hash program.

```
use ashmaize::{Rom, RomGenerationType};

let rom = Rom::new(b"seed", RomGenerationType::FullRandom, 16 * 1_024);
```

## [`hash`]

Now you can use the [`hash`] function to execute a random program against
the [`Rom`] that will generate a Digest.

```
use ashmaize::original::hash;
# use ashmaize::{Rom, RomGenerationType};
# let rom = Rom::new(b"seed", RomGenerationType::FullRandom, 16 * 1_024);

let digest = hash(b"salt", &rom, 8, 256);
# assert_eq!(
#      digest,
#      [39, 200, 35, 132, 250, 151, 62, 36, 195, 37, 55, 6, 49, 113, 39, 252, 116, 112, 101, 81, 253, 131, 10, 219, 152, 249, 52, 72, 200, 130, 140, 178, 31, 219, 153, 103, 73, 154, 110, 196, 245, 10, 65, 203, 223, 3, 64, 51, 154, 179, 86, 174, 136, 107, 27, 89, 29, 235, 97, 95, 230, 159, 207, 58]
# );```

*/

pub mod rom;
pub use crate::rom::{Rom, RomGenerationType};
pub mod original; // original implementation

pub mod b2; // using the blake2 crate
#[cfg(target_os = "macos")]
pub mod metal;
pub mod simd; // using the blake2b_simd crate // GPU-accelerated implementation using Metal
