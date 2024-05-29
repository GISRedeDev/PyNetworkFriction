from pathlib import Path

from net_friction.areas_of_control_matrix import calculate_src_dst_areas_of_control


def test_calculate_src_dst_areas_of_control(make_src_dst_matrix):
    result = calculate_src_dst_areas_of_control(
        make_src_dst_matrix,
        "2024-04-01",
        "2024-04-30",
        Path("tests/test_data/control"),
        6383,
    )
    expected_columns = [
        "from_pcode",
        "to_pcode",
        "src_occupied",
        "dst_occupied",
        "date",
    ]
    assert list(result.columns) == expected_columns
