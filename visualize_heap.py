"""
Provide an ascii-based graphical representation of the heap layout.

Note: Mostly done for x64, other architectures were not throughly tested.
"""

__AUTHOR__ = "hugsy"
__VERSION__ = 0.4
__LICENSE__ = "MIT"

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Tuple
from argparse import ArgumentParser, ArgumentTypeError
import re

import gdb

if TYPE_CHECKING:
    from . import *
    from . import gdb


def fastbin_index(sz):
    return (sz >> 4) - 2 if gef.arch.ptrsize == 8 else (sz >> 3) - 2


def nfastbins():
    return fastbin_index((80 * gef.arch.ptrsize // 4)) - 1


def get_tcache_count():
    version = gef.libc.version
    if version is None:
        raise RuntimeError("Cannot get the libc version")
    if version < (2, 27):
        return 0
    base = gef.heap.base_address
    if not base:
        raise RuntimeError("Failed to get the heap base address. Heap not initialized?")
    count_addr = base + 2 * gef.arch.ptrsize
    count = p8(count_addr) if version < (2, 30) else p16(count_addr)
    return count


@lru_cache(128)
def collect_known_values() -> Dict[int, str]:
    arena = gef.heap.main_arena
    if not arena:
        raise RuntimeError

    version = gef.libc.version
    if not version:
        raise RuntimeError

    result: Dict[int, str] = {}  # format is { 0xaddress : "name" ,}

    # tcache
    if version >= (2, 27):
        tcache_addr = GlibcHeapTcachebinsCommand.find_tcache()
        for i in range(GlibcHeapTcachebinsCommand.TCACHE_MAX_BINS):
            chunk, _ = GlibcHeapTcachebinsCommand.tcachebin(tcache_addr, i)
            j = 0
            while True:
                if chunk is None:
                    break
                sz = (i + 1) * 0x10 + 0x10
                result[chunk.data_address] = f"tcachebins[{i}/{j}] (size={sz:#x})"
                next_chunk_address = chunk.get_fwd_ptr(True)
                if not next_chunk_address:
                    break
                next_chunk = GlibcChunk(next_chunk_address)
                j += 1
                chunk = next_chunk

    # fastbins
    for i in range(nfastbins()):
        chunk = arena.fastbin(i)
        j = 0
        while True:
            if chunk is None:
                break
            result[chunk.data_address] = f"fastbins[{i}/{j}]"
            next_chunk_address = chunk.get_fwd_ptr(True)
            if not next_chunk_address:
                break
            next_chunk = GlibcChunk(next_chunk_address)
            j += 1
            chunk = next_chunk

    # other bins
    for name in ["unorderedbins", "smallbins", "largebins"]:
        i = 0
        while True:
            (fw, bk) = arena.bin(i)
            if (fw, bk) == (0, 0):
                break

            head = GlibcChunk(bk, from_base=True).fwd
            if head == fw:
                continue

            chunk = GlibcChunk(head, from_base=True)
            j = 0
            while True:
                if chunk is None:
                    break
                result[chunk.data_address] = f"{name}[{i}/{j}]"
                next_chunk_address = chunk.get_fwd_ptr(True)
                if not next_chunk_address:
                    break
                next_chunk = GlibcChunk(next_chunk_address, from_base=True)
                j += 1
                chunk = next_chunk

    return result


@lru_cache(128)
def collect_known_ranges() -> List[Tuple[range, str]]:
    result = []
    for entry in gef.memory.maps:
        if not entry.path:
            continue
        path = os.path.basename(entry.path)
        result.append((range(entry.page_start, entry.page_end), path))
    return result


def is_corrupted(chunk: GlibcChunk, arena: GlibcArena) -> bool:
    """Various checks to see if a chunk is corrupted"""

    if chunk.base_address > chunk.data_address:
        return False

    if chunk.base_address > arena.top:
        return True

    if chunk.size == 0:
        return True

    return False


@register
class VisualizeHeapChunksCommand(GenericCommand):
    """Visual helper for glibc heap chunks"""

    _cmdline_ = "visualize-libc-heap-chunks"
    _syntax_ = f"{_cmdline_:s}"
    _aliases_ = [
        "heap-view",
    ]
    _example_ = f"{_cmdline_:s}"

    def __init__(self):
        super().__init__(complete=gdb.COMPLETE_SYMBOL)
        return

    @only_if_gdb_running
    def do_invoke(self, argv):
        # Print out a specific chunk
        parser = ArgumentParser()
        parser.add_argument(
            "address", nargs="*", type=parse_addr, help="Address of the chunk"
        )
        args = parser.parse_args(argv)
        gef_print(f"addr {args.address}")
        return

        # Print out entire heap
        if not gef.heap.main_arena or not gef.heap.base_address:
            err("The heap has not been initialized")
            return

        ptrsize = gef.arch.ptrsize
        arena = gef.heap.main_arena

        colors = ("cyan", "red", "yellow", "blue", "green")
        color_idx = 0
        chunk_idx = 0

        known_ranges = collect_known_ranges()
        known_values = [(range(0x000055555555efa0, 0x000055555555ef8), "SAME")]  # collect_known_values()
        # known_values = collect_known_values()

        for chunk in gef.heap.chunks:
            if is_corrupted(chunk, arena):
                err("Corrupted heap, cannot continue.")
                return

            aggregate_nuls = 0
            base = chunk.base_address

            if base == arena.top:
                gef_print(
                    f"{format_address(base)}    {format_address(gef.memory.read_integer(base))}    "
                    f"{format_address(gef.memory.read_integer(base+ptrsize))}    {Color.colorify(LEFT_ARROW + 'Top Chunk', 'red bold')}"
                )
                break

            # read two pointers at a time
            for current in range(base, base + chunk.size, ptrsize * 2):
                lvalue = gef.memory.read_integer(current)
                rvalue = gef.memory.read_integer(current + ptrsize)
                if lvalue == 0 and rvalue == 0:
                    if current != base and current != (base + chunk.size - ptrsize):
                        # Only aggregate null bytes that are not starting/finishing the chunk
                        aggregate_nuls += 1
                        if aggregate_nuls > 1:
                            continue

                if aggregate_nuls > 2:
                    # If here, we have some aggregated null bytes, print a small thing to mention that
                    gef_print("        ↓        [...]        ↓        [...]        ↓")
                    aggregate_nuls = 0

                # Read the context in a hexdump-like format
                chunk_data = gef.memory.read(
                    current + ptrsize, ptrsize
                ) + gef.memory.read(current, ptrsize)
                hexdump = "".join(
                    map(
                        lambda b: chr(b) if 0x20 <= b < 0x7F else ".",
                        chunk_data,
                    )
                )

                if gef.arch.endianness == Endianness.LITTLE_ENDIAN:
                    hexdump = hexdump[::-1]

                line = f"{format_address(current)}    {Color.colorify(format_address(lvalue), colors[color_idx])}    {Color.colorify(format_address(rvalue), colors[color_idx])}"
                line += f"    {hexdump}"

                # The first entry of the chunk gets added some extra info about the chunk itself
                if current == base:
                    line += f"    Chunk[{chunk_idx:#x}], Size={chunk.usable_size:#x}, Flag={chunk.flags!s}"
                    chunk_idx += 1

                # Populate information for known ranges, if any. Shows where the chunk is located e.g (heap,text)
                lval_known_ranges = []
                for _range, _value in known_ranges:
                    if lvalue in _range:
                        lval_known_ranges.append(Color.redify(_value))

                rval_known_ranges = []
                for _range, _value in known_ranges:
                    if rvalue in _range:
                        rval_known_ranges.append(Color.redify(_value))

                if lval_known_ranges:
                    line += f"    ({', '.join(lval_known_ranges)})|"
                    if rval_known_ranges:
                        line += f"({', '.join(rval_known_ranges)})"
                    else:
                        line += "()"
                else:
                    if rval_known_ranges:
                        line += "    ()|"
                        line += f"({', '.join(rval_known_ranges)})"

                # Populate information from other chunks/bins, if any
                for _range, _value in known_values:
                    if current in _range:
                        line += f"{RIGHT_ARROW}{Color.cyanify(_value)}"

                # deref_line = ""
                # lderefs = dereference_from(current)
                # if len(lderefs) > 2:
                #     deref_line += f"    [{LEFT_ARROW}{lderefs[-1]}]"

                # rderefs = dereference_from(current + ptrsize)
                # if len(rderefs) > 2:
                #     deref_line += f"    [{LEFT_ARROW}{rderefs[-1]}]"
                # line += deref_line

                # All good, print it out
                gef_print(line)

            color_idx = (color_idx + 1) % len(colors)

        return


def parse_addr(string):
    m = re.match(r"(\d+|0x[0-9a-fA-F]+)(?:-(\d+|0x[0-9a-fA-F]+))?$", string)
    if not m:
        raise ArgumentTypeError(
            "'"
            + string
            + "' is not a range of number. Expected forms like '0-5' or '0x10-0x20'"
        )

    addr_ranges = []
    start = m.group(1)
    if start:
        addr_ranges.append(int(start, 0))
    end = m.group(2)
    if end:
        addr_ranges.append(int(end, 0))

    if len(addr_ranges) > 1:
        if addr_ranges[0] > addr_ranges[1]:
            raise ArgumentTypeError(
                "'" + string + "' first number must be smaller than the second one"
            )
    return addr_ranges

VisualizeHeapChunksCommand()
