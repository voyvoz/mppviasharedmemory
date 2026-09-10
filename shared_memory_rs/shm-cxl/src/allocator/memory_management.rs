use crate::allocator::os::OSAlloc;
use crate::core::Cacheline;
use crate::core::host::PeerID;
use crate::cxl::PersistantDataStructs;
use crate::cxl_write_sync;
use allocator_api2::alloc::Allocator;
use anyhow::Result;
use anyhow::anyhow;
use bitflags::bitflags;
use std::alloc::Layout;
use std::os::fd::RawFd;
use std::ptr::NonNull;

/// A memory object represents an allocation made using this library.
/// The Cacheline immediatly preceeding the returned address contains metadata about this object.
///
/// In case this points to CXL memory the cacheline pointed to by addr contains a global view of the metadata
#[derive(Debug)]
pub struct MemoryObject {
    // Location from the perspective of the current process. Never store this raw pointer anywhere!
    addr: NonNull<[u8]>,
    // Metadata
    owner: PeerID,
    flags: Flags,
    layout: Layout, // Size and aligment of this memory block
}

// TODO schauen ob das static ist?
impl MemoryObject {
    /// Checks wether `self` is not an operating system allocation
    pub fn is_local(&self) -> bool {
        return !self.flags.contains(Flags::LOCATION_OS);
    }

    /// Allows changing metadata of `self`
    pub fn update_meta(&self) {
        // Persist changes to metadata
        let meta = self.addr.as_ptr() as *mut Cacheline;
        match self.flags.to_mem_provider_type() {
            MemProviderType::OS => {
                // Nothing needs to be done
            }
            MemProviderType::CXL => {
                // Maybe we should read once again?
                // Maybe we should check the ownership?
                let b = self.owner.to_bytes();
                cxl_write_sync!((meta, 1, b[0]), (meta, 2, b[1]));
            }
            MemProviderType::ACCEL => {}
        }
    }

    /// Required when accessing data of object
    pub fn check_read_allowed(&self) -> bool {
        // Base case -> if we own we can do anything
        return self.owned_by_self();
    }

    /// Required when altering object in any way
    pub fn check_write_allowed(&self) -> bool {
        // Base case -> if we own we can do anything
        return self.owned_by_self();
    }

    /// Checks if `self`has the same owner as the calling process from the perspective of this library
    pub fn owned_by_self(&self) -> bool {
        self.owner.eq(&PeerID::whoami())
    }
}

/// This structure is used to coordinate memory allocations made using this library
/// across different devices that expose memory allocation capabilites.
///
/// TODO cxl and accel should be a list for each device?
pub struct MemoryManager {
    pub os_alloc: Box<dyn Allocator>,
    pub cxl_alloc: Box<dyn Allocator>, // TODO this could be a vec?
    pub accel_alloc: Box<dyn Allocator>, // TODO this could be a vec?
}

#[derive(Debug)]
pub enum MemProviderType {
    OS,
    CXL,
    ACCEL,
}
impl MemProviderType {
    pub fn to_flag(&self) -> Flags {
        match self {
            MemProviderType::OS => Flags::LOCATION_OS,
            MemProviderType::CXL => Flags::LOCATION_CXL,
            MemProviderType::ACCEL => Flags::LOCATION_ACCEL,
        }
    }
}

/// Represents a memory provider which will be mmap'd into this process address space.
/// Thus `file_descriptor` must be compatible with mmap.
#[derive(Debug)]
pub struct ExternalMemProvider {
    kind: MemProviderType,
    file_descriptor: RawFd,
    size: usize,
    mmap_flags: i32,
}

impl ExternalMemProvider {
    /// A CXL shared memory allocator
    pub fn cxl(fd: RawFd, size: usize) -> Self {
        ExternalMemProvider {
            kind: MemProviderType::CXL,
            file_descriptor: fd,
            size,
            mmap_flags: 0,
        }
    }
    /// An allocator for a local Accelerator target
    pub fn accel(fd: RawFd, size: usize) -> Self {
        ExternalMemProvider {
            kind: MemProviderType::ACCEL,
            file_descriptor: fd,
            size,
            mmap_flags: 0,
        }
    }
}

bitflags! {
    /// These values are not stable so dont store them! They are intended to be used directly.
    #[derive(Clone,Copy,Debug)]
    pub struct Flags: u32 {
        // From Where to make the allocation
        const LOCATION_OS = 1;
        const LOCATION_CXL = 1 << 1;
        const LOCATION_ACCEL = 1 << 2;
        // If the memory is intended to be cachable or not
        const HINT_CACHEABLE = 1 << 10;
        const HINT_UNCACHEABLE = 1 << 11;
        // If the memory might retain its values across process restarts
        const HINT_VOLATILE= 1 << 12;
        const HINT_NONVOLATILE = 1 << 13;
    }
}

impl Flags {
    pub fn to_mem_provider_type(&self) -> MemProviderType {
        if self.contains(Flags::LOCATION_CXL) {
            MemProviderType::CXL
        } else if self.contains(Flags::LOCATION_ACCEL) {
            MemProviderType::ACCEL
        } else {
            MemProviderType::OS
        }
    }
}

impl MemoryManager {
    /// Main entrypoint of this library.
    pub fn mm_init(provider: Vec<ExternalMemProvider>) -> Result<Self> {
        // LOCAL mem handling
        // Is there actually anythig we need to do?
        let os_alloc = OSAlloc::create();
        let box_os_alloc = Box::new(os_alloc);
        let mut box_cxl_alloc: Box<dyn Allocator> = Box::new(os_alloc);
        let box_accel_alloc = Box::new(os_alloc);

        let mut result = vec![];

        for p in provider {
            match p.kind {
                MemProviderType::CXL => {
                    // CXL devdax requires mappings of n * 2097152 with n > 0
                    if p.size < 2097152 {
                        return Err(anyhow!(format!("{} < {}", p.size, 2097152)));
                    }
                    // Ensure presence of MAP_SHARED
                    let mapping_flags = libc::MAP_SHARED | p.mmap_flags;
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            p.size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            mapping_flags,
                            p.file_descriptor,
                            0,
                        )
                    };
                    if ptr == libc::MAP_FAILED {
                        return Err(anyhow!(std::io::Error::last_os_error()));
                    }

                    let pds = PersistantDataStructs::init(ptr as *mut Cacheline, p.size);
                    //box_cxl_alloc = Box::new(pds.alloc);

                    result.push(ptr as *mut u8);
                }
                MemProviderType::ACCEL => {
                    // TODO
                }
                _ => {
                    return Err(anyhow!("Invalid provider {:?}", p));
                }
            }
        }

        Ok(Self {
            os_alloc: box_os_alloc,
            cxl_alloc: box_cxl_alloc,
            accel_alloc: box_accel_alloc,
        })
    }

    /// ----------------------
    /// Basic building blocks
    /// ----------------------

    /// Allocate memory
    /// Ownership: The calling process gets implicit ownership of the allocation
    pub fn mm_malloc(&self, layout: Layout, flags: Flags) -> Result<MemoryObject> {
        let res = if flags.contains(Flags::LOCATION_CXL) {
            self.cxl_alloc.allocate(layout).unwrap() // TODO now unwrap
        } else if flags.contains(Flags::LOCATION_ACCEL) {
            self.accel_alloc.allocate(layout).unwrap() // TODO now unwrap
        } else {
            self.os_alloc.allocate(layout).unwrap() // TODO now unwrap
        };

        Ok(MemoryObject {
            addr: res,
            layout,
            owner: PeerID::whoami(),
            flags,
        })
    }

    /// Retruns the `obj` back to the allocator that was used to allocate this MemoryObject.
    pub fn mm_free(&self, obj: &MemoryObject) -> Result<()> {
        if !obj.check_write_allowed() {
            return Err(anyhow!("must be allowed to write target obj"));
        }

        match obj.flags.to_mem_provider_type() {
            MemProviderType::OS => {
                unsafe { self.os_alloc.deallocate(obj.addr.cast(), obj.layout) };
            }
            MemProviderType::CXL => {
                unsafe { self.cxl_alloc.deallocate(obj.addr.cast(), obj.layout) };
            }
            MemProviderType::ACCEL => {
                unsafe { self.accel_alloc.deallocate(obj.addr.cast(), obj.layout) };
            }
        }

        Ok(())
    }

    /// Copies the given memory range to the indicated destination object
    /// m = size of source
    /// n = size of destionation
    /// Requires m <=n. If n < M only n bytes we be changed.
    ///
    /// Ownership:
    ///     Requires that we are allowed to read from source and write to destination
    pub fn mm_copy(&self, source_obj: &MemoryObject, destination_obj: &MemoryObject) -> Result<()> {
        if !source_obj.check_read_allowed() {
            return Err(anyhow!("must be allowed to read from source"));
        }
        if !destination_obj.check_write_allowed() {
            return Err(anyhow!("must be allowed to write to destination"));
        }
        // Depending on the destnation object we gotta choose a fitting write method
        if source_obj.layout.size() <= destination_obj.layout.size() {
            Err(anyhow!(
                "{:?} > {:?} source must fit into destination",
                source_obj.layout.size(),
                destination_obj.layout.size()
            ))
        } else {
            // TODO alignment check?
            //unsafe { memcpy(source_obj.addr.as_ptr() as *mut c_void, destination_obj.addr.as_ptr() as *mut c_void, source_obj.layout.size()) };
            Ok(())
        }
    }

    /// Sets the ownership of the target MemoryObject
    pub fn mm_set_owner(&self, obj: &mut MemoryObject, owner: PeerID) -> Result<()> {
        if !obj.check_write_allowed() {
            return Err(anyhow!("must be allowed to write to destination"));
        }

        obj.owner = owner;
        // In case this MemoryObject is shared we need to flush changes explicitly
        if !obj.is_local() {
            obj.update_meta();
        }

        Ok(())
    }

    /// ----------------------
    /// Combined building blocks
    /// ----------------------

    /// Moves the `source` MemoryObject to the `destination` memory type.
    /// This will first check if enough
    /// This will call `mm_free` on the source object.
    /// Returns a new MemoryObject that owns the memory at the destination.
    pub fn mm_move(
        &self,
        source: &MemoryObject,
        destination_type: MemProviderType,
    ) -> Result<MemoryObject> {
        if !source.check_write_allowed() {
            return Err(anyhow!("must be allowed to write to destination"));
        }

        // Allocate memory
        let destination_obj =
            match self.mm_malloc(source.layout, MemProviderType::to_flag(&destination_type)) {
                Ok(e) => e,
                Err(err) => return Err(err),
            };
        // Copy to destination
        match self.mm_copy(source, &destination_obj) {
            Ok(_) => {}
            Err(err) => return Err(err),
        };

        // Free source
        self.mm_free(source)?;

        Ok(destination_obj)
    }

    pub fn mm_clone(&self, source: &MemoryObject) -> Result<MemoryObject> {
        // malloc
        let destination_obj = match self.mm_malloc(source.layout, source.flags) {
            Ok(e) => e,
            Err(err) => return Err(err),
        };

        // copy
        match self.mm_copy(source, &destination_obj) {
            Ok(_) => {}
            Err(err) => return Err(err),
        };

        Ok(destination_obj)
    }
}
