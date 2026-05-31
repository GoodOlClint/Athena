import Darwin
import Foundation

/// M55 — process memory sampled via Mach `TASK_VM_INFO`.
///
/// `resident` is the classic RSS (`resident_size`): pages mapped into the
/// address space, including the mmap'd model weights. It does NOT count
/// the Metal/GPU unified-memory buffers MLX allocates for the KV cache,
/// prompt cache, and inference activations.
///
/// `physFootprint` is what Activity Monitor's "Memory" column reports
/// (`phys_footprint`): it DOES count those GPU buffers. So during a
/// long-context decode `physFootprint` rises well above `resident` while
/// `resident` stays flat at the weight footprint — surfacing both makes
/// the GPU-transient working set explicit instead of invisible.
///
/// One `TASK_VM_INFO` call yields both. Returns (0, 0) on a Mach failure,
/// preserving the prior degrade-gracefully contract (callers treat 0 as
/// "unknown" and fall back to estimates).
public enum ProcessMemory {
    public static func sample() -> (resident: Int, physFootprint: Int) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size
                / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(
                to: integer_t.self, capacity: Int(count)
            ) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    intPtr, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return (0, 0) }
        return (Int(info.resident_size), Int(info.phys_footprint))
    }

    /// RSS only (`resident_size`) — for callers that want the prior
    /// probe's value unchanged (e.g. the governor reconcile, which
    /// reserves against the mmap'd weight footprint, not transient GPU
    /// buffers).
    public static func residentBytes() -> Int { sample().resident }
}
