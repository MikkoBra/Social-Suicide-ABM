import mesa
from ..system_updates.next_state import (
    stress,
    aversive_internal_state,
    urge_to_escape,
    sigmoid,
    strategy_for_change,
    rk4_step,
)


class StandardAgent(mesa.Agent):
    """
    Default agent in the suicide model.
    """

    def __init__(self, model):
        """
        Initializes the agent with a default stress value.
        """
        super().__init__(model)
        self.stress = 0.5
        self.aversive_internal_state = 0.39
        self.urge_to_escape = 0
        self.suicidal_thought = 0
        self.escape_behavior = 0
        self.external_strat = 0
        self.internal_strat = 0
        self.time = 0
    
    def set_A_params(
            self,
            feedback=6,
            carrying_capacity=0.2,
            stress_weight=3,
            suicidal_thought_weight=0.1,
            escape_behavior_weight=2,
            internal_strat_weight=0.5,
    ):
        return {
            "S": self.stress,
            "T": self.suicidal_thought,
            "X": self.escape_behavior,
            "I": self.internal_strat,
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "S_weight": stress_weight,
            "T_weight": suicidal_thought_weight,
            "X_weight": escape_behavior_weight,
            "I_weight": internal_strat_weight,
        }
    
    def set_U_params(
            self,
            feedback=5,
            aversive_internal_state_weight=3,
    ):
        return {
            "A": self.aversive_internal_state,
            "feedback": feedback,
            "A_weight": aversive_internal_state_weight,
        }
    
    def set_T_params(
            self,
            weight_new=0.8,
            sigmoid_middle = 0.4,
            sigmoid_steepness=100,
    ):
        return {
            "U": self.urge_to_escape,
            "weight_new": weight_new,
            "sig_middle": sigmoid_middle,
            "sig_steepness": sigmoid_steepness,
        }
    
    def set_X_params(
            self,
            weight_new=0.8,
            sigmoid_middle = 0.35,
            sigmoid_steepness=50,
    ):
        return {
            "U": self.urge_to_escape,
            "weight_new": weight_new,
            "sig_middle": sigmoid_middle,
            "sig_steepness": sigmoid_steepness,
        }
    
    def set_E_params(
            self,
            feedback=3,
            carrying_capacity=0.1,
            aversive_internal_state_weight=0.41,
            urge_to_escape_weight=0.6,
    ):
        return {
            "A": self.aversive_internal_state,
            "U": self.urge_to_escape,
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "A_weight": aversive_internal_state_weight,
            "U_weight": urge_to_escape_weight,
        }
    
    def set_I_params(
            self,
            feedback=3,
            carrying_capacity=0.05,
            aversive_internal_state_weight=0.65,
            urge_to_escape_weight=1.05,
    ):
        return {
            "A": self.aversive_internal_state,
            "U": self.urge_to_escape,
            "feedback": feedback,
            "carrying_capacity": carrying_capacity,
            "A_weight": aversive_internal_state_weight,
            "U_weight": urge_to_escape_weight,
        }
    
    def update_agent(self, dt):
        """
        Updates the stress value of the agent.
        """
        new_S = stress(dt=dt, prev_stress=self.stress, mean=0.5, sigma=0.15, reversion=1.0)

        params = self.set_A_params()
        new_A = rk4_step(
            self.aversive_internal_state,
            self.time,
            dt,
            aversive_internal_state,
            params,
        )

        params = self.set_U_params()
        new_U = rk4_step(
            self.urge_to_escape,
            self.time,
            dt,
            urge_to_escape,
            params,
        )

        params = self.set_T_params()
        new_T = sigmoid(self.suicidal_thought, self.time, params)

        params = self.set_X_params()
        new_X = sigmoid(self.suicidal_thought, self.time, params)

        params = self.set_E_params()
        new_E = rk4_step(
            self.external_strat,
            self.time,
            dt,
            strategy_for_change,
            params,
        )

        params = self.set_I_params()
        new_I = rk4_step(
            self.internal_strat,
            self.time,
            dt,
            strategy_for_change,
            params,
        )

        self.stress = new_S
        self.aversive_internal_state = new_A
        self.urge_to_escape = new_U
        self.suicidal_thought = new_T
        self.escape_behavior = new_X
        self.external_strat = new_E
        self.internal_strat = new_I
        self.time += dt
