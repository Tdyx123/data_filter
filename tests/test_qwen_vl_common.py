from qwen3_vl_groot.data import BridgeMetadata as LegacyBridgeMetadata
from qwen3_vl_groot.normalization import QuantileStats as LegacyQuantileStats
from qwen3_vl_groot.schedules import LoraUpdateSchedule as LegacyLoraUpdateSchedule


def test_common_mechanisms_keep_legacy_groot_imports_as_aliases():
    from qwen_vl_common.data import BridgeMetadata
    from qwen_vl_common.normalization import QuantileStats
    from qwen_vl_common.schedules import LoraUpdateSchedule

    assert BridgeMetadata is LegacyBridgeMetadata
    assert QuantileStats is LegacyQuantileStats
    assert LoraUpdateSchedule is LegacyLoraUpdateSchedule


def test_common_backbone_loader_keeps_legacy_groot_import_as_alias():
    from qwen3_vl_groot.config import backbone_contract as legacy_contract
    from qwen3_vl_groot.modeling import (
        inspect_qwen_config as legacy_inspector,
        load_qwen_backbone as legacy_loader,
    )
    from qwen_vl_common.backbone import inspect_qwen_config, load_qwen_backbone
    from qwen_vl_common.contracts import backbone_contract

    assert load_qwen_backbone is legacy_loader
    assert inspect_qwen_config is legacy_inspector
    assert backbone_contract is legacy_contract
