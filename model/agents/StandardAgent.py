import mesa
import numpy as np
from ..system_updates.next_state import (
    stress,
    aversive_internal_state,
    urge_to_escape,
    sigmoid,
    strategy_for_change,
    rk4_step,
)
SOCIAL_WEIGHT_IDX = 1


class StandardAgent(mesa.Agent):
    """
    Default agent in the suicide model.
    """

    def __init__(self, model):
        """
        Initializes the agent with a default stress value.
        """
        super().__init__(model)
        # Initial values
        self.stress = 0.5
        self.aversive_internal_state = 0.39
        self.urge_to_escape = 0
        self.suicidal_thought = 0
        self.escape_behavior = 0
        self.external_strat = 0
        self.internal_strat = 0
        self.total_time = 0

        # Stress parameters
        self.S_mean = 0.5
        self.S_sigma = 0.15
        self.S_reversion = 1.0
        self.S_E_weight = 3.0

        # Aversive internal state parameters
        self.A_feedback = 6
        self.A_carrying_capacity = 0.2
        self.A_S_weight = 3
        self.A_T_weight = 0.1
        self.A_X_weight = 2
        self.A_I_weight = 0.5
        self.A_F_weight = 7
        self.A_B_weight = 5

        # Urge to escape parameters
        self.U_feedback = 5
        self.U_A_weight = 3

        # Suicidal thought parameters
        self.T_weight_new = 0.8
        self.T_sig_middle = 0.4
        self.T_sig_steepness = 100

        # Escape behavior parameters
        self.X_weight_new = 0.8
        self.X_sig_middle = 0.35
        self.X_sig_steepness = 50

        # External strategy parameters
        self.E_feedback=3
        self.E_carrying_capacity=0.1
        self.E_A_weight=0.41
        self.E_U_weight=0.6

        # Internal strategy parameters
        self.I_feedback=3
        self.I_carrying_capacity=0.05
        self.I_A_weight=0.65
        self.I_U_weight=1.05

    def set_friends(self, n=2):
        self.friends = self.set_social_connections(n)
        self.num_friends = n

    def set_bullies(self, n=1):
        self.bullies = self.set_social_connections(n)
        self.num_bullies = n
    
    def set_social_connections(
            self,
            n
    ):
        other_agents = [agent.unique_id 
                        for agent in self.model.agents 
                        if agent.unique_id != self.unique_id]
        n = min(n, len(other_agents))
        agent_IDs = np.random.choice(other_agents, size=n)

        # Take n random samples from N(0.5, 0.15)
        # (roughly between 0 and 1), then clip
        weights = np.random.normal(loc=0.5, scale=0.15, size=n)
        weights = np.clip(weights, 0, 1)

        # Make 2D array
        connections = np.column_stack((agent_IDs, weights))
        return connections

    def saturated_mean_social_influence(self, connections, k=5):
        """
        Calculates social influence using the mean weight
        of the agent's social connections. Saturates by
        multiplying with f(n) = n/(k+n), so that f(n) =
        1 as n -> inf, and f(n) = 1/2 at k = n.
        """
        total = 0
        for agent_info in connections:
            total += agent_info[SOCIAL_WEIGHT_IDX]
        n = len(connections)
        return (total/n) * (n/(k+n))
    
    def get_A_params(
            self,
    ):
        """
        Gathers all aversive internal state parameters into
        a dictionary.
        """
        return {
            "S": self.stress,
            "T": self.suicidal_thought,
            "X": self.escape_behavior,
            "I": self.internal_strat,
            "F": self.saturated_mean_social_influence(self.friends),
            "B": self.saturated_mean_social_influence(self.bullies),
            "feedback": self.A_feedback,
            "carrying_capacity": self.A_carrying_capacity,
            "S_weight": self.A_S_weight,
            "T_weight": self.A_T_weight,
            "X_weight": self.A_X_weight,
            "I_weight": self.A_I_weight,
            "F_weight": self.A_F_weight,
            "B_weight": self.A_B_weight,
        }
    
    def get_U_params(
            self,
    ):
        """
        Gathers all urge to escape parameters into a dictionary.
        """
        return {
            "A": self.aversive_internal_state,
            "feedback": self.U_feedback,
            "A_weight": self.U_A_weight,
        }
    
    def get_T_params(
            self,
    ):
        """
        Gathers all suicidal thought parameters into a dictionary.
        """
        return {
            "U": self.urge_to_escape,
            "weight_new": self.T_weight_new,
            "sig_middle": self.T_sig_middle,
            "sig_steepness": self.T_sig_steepness,
        }
    
    def get_X_params(
            self,
    ):
        """
        Gathers all escape behavior parameters into a dictionary.
        """
        return {
            "U": self.urge_to_escape,
            "weight_new": self.X_weight_new,
            "sig_middle": self.X_sig_middle,
            "sig_steepness": self.X_sig_steepness,
        }
    
    def get_E_params(
            self,
    ):
        """
        Gathers all external strategy parameters into a dictionary.
        """
        return {
            "A": self.aversive_internal_state,
            "U": self.urge_to_escape,
            "feedback": self.E_feedback,
            "carrying_capacity": self.E_carrying_capacity,
            "A_weight": self.E_A_weight,
            "U_weight": self.E_U_weight,
        }
    
    def get_I_params(
            self,
    ):
        """
        Gathers all internal strategy parameters into a dictionary.
        """
        return {
            "A": self.aversive_internal_state,
            "U": self.urge_to_escape,
            "feedback": self.I_feedback,
            "carrying_capacity": self.I_carrying_capacity,
            "A_weight": self.I_A_weight,
            "U_weight": self.I_U_weight,
        }
    
    def update_agent(self, dt):
        """
        Updates the agent over timestep dt.
        """
        # Update stress
        new_S = stress(
            dt=dt,
            prev_stress=self.stress,
            prev_E=self.external_strat,
            mean=self.S_mean,
            sigma=self.S_sigma,
            reversion=self.S_reversion,
            prev_E_weight=self.S_E_weight)

        # Update aversive internal state
        params = self.get_A_params()
        new_A = rk4_step(
            self.aversive_internal_state,
            self.total_time,
            dt,
            aversive_internal_state,
            params,
        )

        # Update urge to escape
        params = self.get_U_params()
        new_U = rk4_step(
            self.urge_to_escape,
            self.total_time,
            dt,
            urge_to_escape,
            params,
        )

        # Update suicidal thought
        params = self.get_T_params()
        new_T = sigmoid(self.suicidal_thought, self.total_time, params)

        # Update escape behavior
        params = self.get_X_params()
        new_X = sigmoid(self.escape_behavior, self.total_time, params)

        # Update external strategy
        params = self.get_E_params()
        new_E = rk4_step(
            self.external_strat,
            self.total_time,
            dt,
            strategy_for_change,
            params,
        )

        # Update internal strategy
        params = self.get_I_params()
        new_I = rk4_step(
            self.internal_strat,
            self.total_time,
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
        self.total_time += dt
