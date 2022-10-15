import gc
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, List, Sequence

import torch


def tensor_hash(x: torch.Tensor) -> int:
    return hash(pickle.dumps(x.detach().cpu().numpy()))


@dataclass(order=True)
class Duplicate:
    size: int
    indices: List[int]
    tensors: List[torch.Tensor] = field(compare=False, repr=False, hash=False)

    @property
    def count(self) -> int:
        return len(self.indices)


# TODO: this does not consider possible hash collisions
def find_dups(tensors: Sequence[torch.Tensor]) -> List[Duplicate]:
    idx_by_hash: DefaultDict[int, List[int]] = defaultdict(lambda: [])

    for i, t in enumerate(tensors):
        idx_by_hash[tensor_hash(t)].append(i)

    dups: List[Duplicate] = []
    for idxs in idx_by_hash.values():
        if len(idxs) < 2:
            continue
        ts = [tensors[x] for x in idxs]
        dups.append(Duplicate(size=ts[0].element_size() * ts[0].nelement(), indices=idxs, tensors=ts))

    dups.sort(reverse=True)
    return dups


def find_dups_in_memory() -> List[Duplicate]:
    return find_dups([obj for obj in gc.get_objects() if torch.is_tensor(obj)])  # type: ignore[no-untyped-call]


def report_dups_in_memory(logger: Any) -> None:
    dups = find_dups_in_memory()
    if not dups:
        logger.debug("No duplicate torch tensors found in memory.")
    waste = 0
    total = 0
    logger.debug("Duplicate torch tensors:")
    for dup in dups:
        logger.debug("{}", dup)
        waste += dup.size * (dup.count - 1)
        total += dup.size * dup.count

    if total == 0:
        return

    mib = 1024 * 1024
    logger.debug("Total tensor bytes: {} ({} MiB)", total, total / mib)
    logger.debug("Total wasted bytes: {} ({} MiB, {:.2f}%)", waste, waste / mib, waste / total * 100.0)
