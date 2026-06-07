from .mesh_block import BoundaryType, MeshBlock


def copy_boundary_values(
    source_block: MeshBlock,
    source_boundary: BoundaryType,
    target_block: MeshBlock,
    target_boundary: BoundaryType,
) -> None:
    """
    Copy boundary values from one mesh block to another.

    This function extracts boundary values from the source block and applies
    them to the target block, with optional scaling and offset.

    Args:
        source_block (MeshBlock): Source mesh block to copy from
        target_block (MeshBlock): Target mesh block to copy to
        source_boundary (BoundaryType): Boundary of source block to copy from
        target_boundary (BoundaryType): Boundary of target block to copy to

    Raises:
        ValueError: If the boundaries have incompatible dimensions
        TypeError: If any arguments have incorrect types

    Example:
        # Copy left boundary values from block1 to right boundary of block2
        copy_boundary_values(block1, block2, BoundaryType.LEFT, BoundaryType.RIGHT)
    """

    source_values = source_block.get_boundary_values(source_boundary)
    target_block.set_boundary_values(target_boundary, source_values)


def copy_boundary_gradients(
    source_block: MeshBlock,
    source_boundary: BoundaryType,
    target_block: MeshBlock,
    target_boundary: BoundaryType,
    scaling_factor: float = 1.0,
) -> None:
    """
    Copy boundary gradients from one mesh block to another.

    This function extracts boundary gradients from the source block and applies
    them to the target block, with optional scaling. The gradients are applied
    by updating boundary values to achieve the desired gradients.

    Args:
        source_block (MeshBlock): Source mesh block to copy from
        target_block (MeshBlock): Target mesh block to copy to
        source_boundary (BoundaryType): Boundary of source block to copy from
        target_boundary (BoundaryType): Boundary of target block to copy to
        scaling_factor (float): Factor to scale the copied gradients (default: 1.0)

    Raises:
        ValueError: If the boundaries have incompatible dimensions or mesh blocks
            are too small
        TypeError: If any arguments have incorrect types

    Example:
        # Copy left boundary gradients from block1 to right boundary of block2
        copy_boundary_gradients(block1, block2, BoundaryType.LEFT, BoundaryType.RIGHT)
    """

    source_gradients = source_block.get_boundary_gradients(source_boundary)
    scaled_gradients = source_gradients * scaling_factor
    target_block.set_boundary_gradients(target_boundary, scaled_gradients)
