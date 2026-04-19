import random
import torch
from torch.utils.data import Dataset, DataLoader

from karel.world import World

IMG_FEAT = 5184
IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))


def grid_desc_to_tensor(grid_desc):
    grid = torch.Tensor(IMG_FEAT).fill_(0)
    grid.index_fill_(0, grid_desc.long(), 1)
    grid = grid.view(IMG_SIZE)
    return grid


class KarelDataset(Dataset):
    def __init__(self, dataset_dict):
        self.sources = dataset_dict["sources"]
        self.targets = dataset_dict["targets"]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]


def get_collate_fn(
    start_idx, end_idx, pad_idx, nb_ios, shuffle=True, load_worlds=False
):
    def collate_fn(batch):
        grid_descriptions = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        inp_grids = []
        out_grids = []
        inp_test_grids = []
        out_test_grids = []

        inp_worlds = []
        out_worlds = []
        inp_test_worlds = []
        out_test_worlds = []

        for sample in grid_descriptions:
            sample_copy = list(sample)
            if shuffle:
                random.shuffle(sample_copy)

            sample_inp_grids = []
            sample_out_grids = []
            sample_test_inp_grids = []
            sample_test_out_grids = []

            sample_inp_worlds = []
            sample_out_worlds = []
            sample_test_inp_worlds = []
            sample_test_out_worlds = []

            for example in sample_copy[:nb_ios]:
                inp_grid_desc, out_grid_desc = example[:2]
                inp_grid = grid_desc_to_tensor(inp_grid_desc)
                out_grid = grid_desc_to_tensor(out_grid_desc)

                sample_inp_grids.append(inp_grid)
                sample_out_grids.append(out_grid)

                if load_worlds:
                    sample_inp_worlds.append(World.fromPytorchTensor(inp_grid))
                    sample_out_worlds.append(World.fromPytorchTensor(out_grid))

            for example in sample_copy[nb_ios:]:
                inp_grid_desc, out_grid_desc = example[:2]
                inp_grid = grid_desc_to_tensor(inp_grid_desc)
                out_grid = grid_desc_to_tensor(out_grid_desc)

                sample_test_inp_grids.append(inp_grid)
                sample_test_out_grids.append(out_grid)

                if load_worlds:
                    sample_test_inp_worlds.append(World.fromPytorchTensor(inp_grid))
                    sample_test_out_worlds.append(World.fromPytorchTensor(out_grid))

            inp_grids.append(torch.stack(sample_inp_grids, 0))
            out_grids.append(torch.stack(sample_out_grids, 0))

            if load_worlds:
                inp_worlds.append(sample_inp_worlds)
                out_worlds.append(sample_out_worlds)

            if sample_test_inp_grids:
                inp_test_grids.append(torch.stack(sample_test_inp_grids, 0))
                out_test_grids.append(torch.stack(sample_test_out_grids, 0))

                if load_worlds:
                    inp_test_worlds.append(sample_test_inp_worlds)
                    out_test_worlds.append(sample_test_out_worlds)

        inp_grids = torch.stack(inp_grids, 0)
        out_grids = torch.stack(out_grids, 0)
        if inp_test_grids:
            inp_test_grids = torch.stack(inp_test_grids, 0)
            out_test_grids = torch.stack(out_test_grids, 0)

        lines = [[start_idx] + line for line in targets]
        lens = [len(line) for line in lines]
        max_len = max(lens)

        input_lines = [
            line[: max_len - 1] + [pad_idx] * (max_len - len(line[: max_len - 1]) - 1)
            for line in lines
        ]
        output_lines = [line[1:] + [pad_idx] * (max_len - len(line)) for line in lines]

        in_tgt_seq = torch.LongTensor(input_lines)
        out_tgt_seq = torch.LongTensor(output_lines)

        if load_worlds:
            return (
                inp_grids,
                out_grids,
                in_tgt_seq,
                input_lines,
                out_tgt_seq,
                targets,
                inp_test_grids,
                out_test_grids,
                inp_worlds,
                out_worlds,
                inp_test_worlds,
                out_test_worlds,
            )
        else:
            return (
                inp_grids,
                out_grids,
                in_tgt_seq,
                input_lines,
                out_tgt_seq,
                targets,
                inp_test_grids,
                out_test_grids,
            )

    return collate_fn


class BatchSampler:
    def __init__(self, targets, batch_size, shuffle_batches=True):
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches

        indices = list(range(len(targets)))
        indices.sort(key=lambda x: len(targets[x]) / 5, reverse=True)

        self.grouped_indices = [
            indices[pos : pos + batch_size]
            for pos in range(0, len(indices), batch_size)
        ]

    def __iter__(self):
        grouped_indices = list(self.grouped_indices)

        if self.shuffle_batches and len(grouped_indices) > 1:
            to_shuffle = grouped_indices[:-1]
            random.shuffle(to_shuffle)
            grouped_indices[:-1] = to_shuffle

        for batch in grouped_indices:
            yield batch

    def __len__(self):
        return len(self.grouped_indices)


def create_dataloader(
    dataset_dict,
    batch_size,
    start_idx,
    end_idx,
    pad_idx,
    nb_ios,
    shuffle_batches=True,
    num_workers=16,
    load_worlds=False,
):
    dataset = KarelDataset(dataset_dict)
    collate_fn = get_collate_fn(
        start_idx, end_idx, pad_idx, nb_ios, shuffle=True, load_worlds=load_worlds
    )
    batch_sampler = BatchSampler(
        dataset_dict["targets"], batch_size, shuffle_batches
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    return loader
