from __future__ import annotations

from tests.algos.rlgames_sapg._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

from tests.algos.rlgames_sapg.source_network_harness import load_source_owner_contract


def test_source_owner_provenance_and_shape_contract() -> None:
    contract = load_source_owner_contract()

    assert contract["source_head"] == "2a9917533bfea70419ed2667a511d7238e5b3abc"
    assert contract["source_rl_games_tree"] == "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
    assert contract["train_owner_blob"] == "f363d05d4a24b190b7837703b93270d8f3fe9a9c"
    assert contract["train_owner_sha256"] == (
        "04f30820094b062412541764b3feeb1492097e75afe5ad0df3fd0e2853496d34"
    )
    assert contract["task_owner_blob"] == "6469d46867081b70edaa589dcb31c7090b64d45e"
    assert contract["task_owner_sha256"] == (
        "9d2bf514f75cc8c72b20da1e8ec971163bbd4cbdf6fc74812aa4a509340acb5e"
    )
    assert contract["coefficient_ids"] == [50, 40, 30, 20, 10, 0]
    assert contract["num_envs"] == 24576
    assert contract["block_size"] == 4096
    assert contract["num_blocks"] == 6
    assert contract["actor_obs"] == 140
    assert contract["actor_carrier"] == 141
    assert contract["actor_embedded_input"] == 172
    assert contract["critic_obs"] == 162
    assert contract["central_carrier"] == 163
    assert contract["central_embedded_input"] == 194
    assert contract["actions"] == 29
    assert contract["actor_embedding_shape"] == [6, 32]
    assert contract["central_embedding_shape"] == [6, 32]
    assert contract["conditional_sigma_shape"] == [6, 29]
    assert contract["actor_architecture"] == ["lstm", "layer_norm", "mlp"]
    assert contract["central_architecture"] == ["mlp"]
