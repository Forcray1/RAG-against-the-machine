from student.models import MinimalSource


def calculate_overlap(r: MinimalSource, g: MinimalSource) -> float:
    """
    Calculate the intersection betzween retrieved text and ground truth
    """
    if r.file_path != g.file_path:
        return 0.0

    # found common zone
    start_overlap = max(r.first_character_index, g.first_character_index)
    end_overlap = min(r.last_character_index, g.last_character_index)

    intersection = max(0, end_overlap - start_overlap)
    taille_g = max(1, g.last_character_index - g.first_character_index)

    return intersection / taille_g
