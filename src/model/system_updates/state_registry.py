from model.states.SleepState import SleepState
from model.states.MorningState import MorningState
from model.states.CommuteState import CommuteState
from model.states.WorkState import WorkState
from model.states.HomeState import HomeState

STATE_REGISTRY = {}

def get_state(name):
    return STATE_REGISTRY[name]

def register_all_states():
    STATE_REGISTRY.update({
        "sleep": SleepState,
        "morning": MorningState,
        "commute": CommuteState,
        "work": WorkState,
        "home": HomeState
    })
