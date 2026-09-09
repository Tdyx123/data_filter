"""Bridge CLI configuration and replay for original-command deviation."""
import copy

import pytest

from cocore_bridge_v2 import cli
from cocore_bridge_v2 import config as bridge_config

SETTINGS = {
    'action_source': 'original_command',
    'action_semantics': 'delta_from_observed_position',
    'action_scale': [1.0, 1.0, 1.0],
    'alignment_confirmed': True,
}
FLAGS = [
    '--execution-action-source', 'original_command',
    '--execution-action-semantics', 'delta_from_observed_position',
    '--execution-action-scale', '1', '1', '1',
    '--execution-alignment-confirmed',
]


def arguments(command='run'):
    return [command, '--relation', 'sequence', '--relation-weight', '1',
            '--output-dir', '/tmp/bridge-execution-test']


@pytest.mark.parametrize('command,stage', [
    ('scan', 'scan_stage'), ('encode', 'encode_stage'),
    ('build-graph', 'graph_stage'), ('select', 'select_stage'),
    ('run', 'run_pipeline'), ('validate', 'validate_output'),
])
def test_every_stage_receives_execution_settings(monkeypatch, command, stage):
    class ReachedStage(Exception):
        pass

    def capture(*args, **kwargs):
        config = kwargs['config'] if command == 'validate' else args[0]
        assert config['action_execution_deviation'] == SETTINGS
        raise ReachedStage

    monkeypatch.setattr(cli, stage, capture)
    monkeypatch.setattr(cli, 'validate_bridge_dataset', lambda path: None)
    with pytest.raises(ReachedStage):
        cli.main(arguments(command) + FLAGS)


def test_yaml_preserved_and_cli_replaces_whole_block(monkeypatch):
    base = bridge_config._load_base_config()
    base['action_execution_deviation'] = {**SETTINGS, 'action_scale': [2, 3, 4]}
    monkeypatch.setattr(bridge_config, '_load_base_config', lambda: copy.deepcopy(base))
    kwargs = dict(relation='sequence', relation_weight=1)
    result = bridge_config.build_config(**kwargs)
    assert result['action_execution_deviation']['action_scale'] == [2, 3, 4]
    assert result['reliability_metrics'] == list(bridge_config.DEFAULT_RELIABILITY_METRICS)
    result = bridge_config.build_config(**kwargs, action_execution_deviation=SETTINGS)
    assert result['action_execution_deviation'] == SETTINGS
    with pytest.raises(ValueError, match='action_execution_deviation'):
        bridge_config.build_config(**kwargs, action_execution_deviation={'action_scale': [1, 1, 1]})


@pytest.mark.parametrize('start,end', [(0, 2), (2, 4), (4, 8), (8, 9)])
def test_partial_cli_is_not_completed_from_yaml(monkeypatch, start, end):
    base = bridge_config._load_base_config()
    base['action_execution_deviation'] = SETTINGS
    monkeypatch.setattr(bridge_config, '_load_base_config', lambda: copy.deepcopy(base))
    with pytest.raises(ValueError, match='action_execution_deviation'):
        cli.main(arguments() + FLAGS[:start] + FLAGS[end:])


@pytest.mark.parametrize('flag,value', [
    ('--execution-action-source', 'state_difference'),
    ('--execution-action-semantics', 'velocity'),
    ('--execution-action-scale', '1 2'),
    ('--execution-action-scale', '1 2 3 4'),
])
def test_parser_rejects_invalid_choices_and_dimensions(flag, value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments() + [flag] + value.split())


@pytest.mark.parametrize('scale', ['0', '-1', 'nan', 'inf'])
def test_cli_rejects_invalid_scale(scale):
    flags = list(FLAGS)
    flags[5] = scale
    with pytest.raises(ValueError, match='action_scale'):
        cli.main(arguments() + flags)


def test_fusion_requires_explicit_configuration():
    with pytest.raises(ValueError, match='requires explicit action_execution_deviation'):
        cli.main(arguments() + ['--reliability-metrics', 'support', 'low_action_execution_deviation'])


def test_cli_execution_roundtrip_and_scale_cache_rejection(tmp_path, monkeypatch):
    from test_cocore_action_execution_deviation_integration import config_for
    from test_cocore_pipeline import CocoreVisualEncoder
    from cocore.pipeline import run_pipeline, validate_output

    synthetic = config_for(tmp_path, profile='bridge_v2')
    synthetic.pop('action_execution_deviation')
    monkeypatch.setattr(bridge_config, '_load_base_config', lambda: copy.deepcopy(synthetic))
    monkeypatch.setattr(cli, 'validate_bridge_dataset', lambda path: None)
    results = []

    def run(config, **kwargs):
        result = run_pipeline(config, visual_encoder=CocoreVisualEncoder(), **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(cli, 'run_pipeline', run)
    common = ['--relation', 'sequence', '--relation-weight', '1',
              '--dataset-path', synthetic['dataset']['path'],
              '--selection-ratio', '0.5', '--support-k', '2',
              '--reliability-metrics', 'support', 'low_action_execution_deviation']
    root = tmp_path / 'output'
    cli.main(['run', '--output-dir', str(root)] + common + FLAGS)
    validated = []

    def validate(path, *, config):
        result = validate_output(path, config=config)
        validated.append(result)
        return result

    monkeypatch.setattr(cli, 'validate_output', validate)
    cli.main(['validate', '--output-dir', str(results[0])] + common + FLAGS)
    assert validated[0]['status'] == 'valid'
    changed = list(FLAGS)
    changed[5] = '2'
    with pytest.raises(ValueError, match='execution|contract|config|fingerprint'):
        cli.main(['validate', '--output-dir', str(results[0])] + common + changed)
    with pytest.raises(FileExistsError, match='incompatible'):
        cli.main(['run', '--output-dir', str(root)] + common + changed)
