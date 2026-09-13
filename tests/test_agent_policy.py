from dataclasses import FrozenInstanceError

import pytest

from peterbot.agent_policy import AgentPolicy, PolicyDenied, Principal


def policy(**kwargs):
    return AgentPolicy(allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}), **kwargs)


def test_empty_policy_and_dms_fail_closed_even_for_matching_roles():
    for principal in (Principal(10, 1, 20, (100,)), Principal(None, 1, 20, (100,))):
        with pytest.raises(PolicyDenied):
            AgentPolicy(officer_role_ids=frozenset({100})).require_admission(principal)
    with pytest.raises(PolicyDenied):
        policy().require_admission(Principal(None, 1, 20, (100,)))


def test_roles_are_ids_and_authority_is_immutable_and_guild_bound():
    officer = Principal(10, 1, 20, (100,))
    assert policy().is_officer(officer)
    assert not policy().is_officer(Principal(11, 1, 20, (100,)))
    assert not policy().is_officer(Principal(10, 1, 20, (999,)))
    with pytest.raises(FrozenInstanceError):
        officer.role_ids = (999,)
    with pytest.raises(ValueError):
        Principal(10, 1, 20, ("president",))


def test_owner_is_separate_from_officer_and_does_not_bypass_pilot():
    owner = Principal(10, 1, 20)
    configured = policy(owner_user_ids=frozenset({1}))
    assert configured.is_owner(owner)
    assert not configured.is_officer(owner)
    with pytest.raises(PolicyDenied):
        configured.require_admission(owner)
    with pytest.raises(PolicyDenied):
        configured.require_memory_access(owner, "club", write=True)


def test_member_pilot_switch_and_memory_permissions():
    member = Principal(10, 2, 20)
    with pytest.raises(PolicyDenied):
        policy().require_admission(member)
    policy(officer_only=False).require_admission(member)
    policy().require_memory_access(member, "personal", write=True)
    policy().require_memory_access(member, "club")
    with pytest.raises(PolicyDenied):
        policy().require_memory_access(member, "club", write=True)
    with pytest.raises(PolicyDenied):
        policy().require_memory_access(member, "officer")


@pytest.mark.parametrize("value", [True, 0, -1, "10", 2**63])
def test_invalid_ids_fail(value):
    with pytest.raises(ValueError):
        Principal(value, 1, 20)
    with pytest.raises(ValueError):
        AgentPolicy(allowed_guild_ids=frozenset({value}))


def test_policy_and_principal_do_not_accept_mutable_collections():
    with pytest.raises(ValueError):
        AgentPolicy(allowed_guild_ids={10})
    with pytest.raises(ValueError):
        Principal(10, 1, 20, [100])
