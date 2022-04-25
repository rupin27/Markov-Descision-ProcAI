from collections import defaultdict


class MDP():
    """Class for representing a Gridworld MDP.

    States are represented as (x, y) tuples, starting at (1, 1).  It is assumed that there are
    four actions from each state (up, down, left, right), and that moving into a wall results in
    no change of state.  The transition model is specified by the arguments to the constructor (with
    probability prob_forw, the agent moves in the intended direction. It veers to either side with
    probability of (1-prob_forw)/2 each.  If the agent runs into a wall, it stays in place.

    """

    def __init__(self, num_rows, num_cols, rewards, terminals, prob_forw, reward_default=0.0):
        """
        Constructor for this MDP.

        Args:
            num_rows: the number of rows in the grid
            num_cols: the number of columns in the grid
            rewards: a dictionary specifying the reward function, with (x, y) state tuples as keys,
                and rewards amounts as values.  If states are not specified, their reward is assumed
                to be equal to the reward_default defined below
            terminals: a list of state (x, y) tuples specifying which states are terminal
            prob_forw: probability of going in the intended direction
            reward_default: reward for any state not specified in rewards
        """
        self.nrows = num_rows
        self.ncols = num_cols
        self.states = []
        for i in range(num_cols):
            for j in range(num_rows):
                self.states.append((i+1, j+1))
        self.rewards = rewards
        self.terminals = terminals
        self.prob_forw = prob_forw
        self.prob_side = (1.0 - prob_forw)/2
        self.reward_def = reward_default
        self.actions = ['up', 'right', 'down', 'left']

    def get_states(self):
        """Return a list of all states as (x, y) tuples."""
        return self.states

    def get_actions(self, state):
        """Return list of possible actions from each state."""
        return self.actions

    def get_successor_probs(self, state, action):
        """Returns a dictionary mapping possible successor states to their transition probabilities
        for the given state and action.
        """
        if self.is_terminal(state):
            return {}  # we cant move from terminal state since we end

        x, y = state
        succ_up = (x, min(self.nrows, y+1))
        succ_right = (min(self.ncols, x+1), y)
        succ_down = (x, max(1, y-1))
        succ_left = (max(1, x-1), y)

        succ__prob = defaultdict(float)
        if action == 'up':
            succ__prob[succ_up] = self.prob_forw
            succ__prob[succ_right] += self.prob_side
            succ__prob[succ_left] += self.prob_side
        elif action == 'right':
            succ__prob[succ_right] = self.prob_forw
            succ__prob[succ_up] += self.prob_side
            succ__prob[succ_down] += self.prob_side
        elif action == 'down':
            succ__prob[succ_down] = self.prob_forw
            succ__prob[succ_right] += self.prob_side
            succ__prob[succ_left] += self.prob_side
        elif action == 'left':
            succ__prob[succ_left] = self.prob_forw
            succ__prob[succ_up] += self.prob_side
            succ__prob[succ_down] += self.prob_side
        return succ__prob

    def get_reward(self, state):
        """Get the reward for the state, return default if not specified in the constructor."""
        return self.rewards.get(state, self.reward_def)

    def is_terminal(self, state):
        """Returns True if the given state is a terminal state."""
        return state in self.terminals


def value_iteration(mdp, gamma, epsilon):
    """Calculate the utilities for the states of an MDP.

    Args:
        mdp: An instance of the MDP class defined above, describing the environment
        gamma: the discount factor
        epsilon: the change threshold to use when determining convergence.  The function returns
            when none of the states have a utility whose change from the previous iteration is more
            than epsilon

    Returns:
        A dictionary, with state (x, y) tuples as keys, and converged utilities as values.
    """
    utilities = {}  # (x, y) -> util
    numTermsts = 0

    for state in mdp.get_states():
        if mdp.is_terminal(state):
            numTermsts += 1
        if state not in utilities:
            utilities[state] = mdp.get_reward(state)
    count = 0
    notCnvrged = True
    while (notCnvrged):
        '''
        print("Iteration: {}".format(count))
        policy = derive_policy(mdp, utilities)
        print(ascii_grid_policy(policy))
        print(ascii_grid_utils(utilities))
        '''
        count += 1
        new_utilities = {}
        statesBelowEpsilon = 0
        states = mdp.get_states()
        for state in states:
            if (not mdp.is_terminal(state)):
                maxPU = -100000 #maximum probable utility  
                for action in mdp.get_actions(state): 
                    currActionPUsum = getUtilProbSum(mdp, state, action, utilities) #current action probable utility sum 
                    maxPU = max(maxPU, currActionPUsum) 
                newUtil = mdp.get_reward(state) + (gamma * maxPU)
                if (abs(newUtil - utilities[state]) < epsilon):
                    statesBelowEpsilon += 1
                new_utilities[state] = newUtil
            else:
                new_utilities[state] = mdp.get_reward(state)
        utilities = new_utilities
        if (statesBelowEpsilon == len(states) - numTermsts):
            notCnvrged = False       

    return utilities

def getUtilProbSum(mdp, state, action, utilities):
    sum = 0
    stateProbDict = mdp.get_successor_probs(state, action)
    for state in stateProbDict: 
        probability = stateProbDict[state]
        sum += (probability * utilities[state])
    return sum


def derive_policy(mdp, utility):
    """Create a policy from an MDP and a set of utilities for each state.

    Args:
        mdp: An instance of the MDP class defined above, describing the environment
        utility: A dictionary mapping state (x, y) tuples to a utility value (perhaps calculated
            from value iteration)

    Returns:
        policy: A dictionary mapping state (x, y) tuples to the optimal action for that state (one
            of 'up', 'down', 'left', 'right', or None for terminal states)
    """
    policy = {}
    for state in mdp.get_states():
        if mdp.is_terminal(state):
            policy[state] = None
        else:
            best_action = None
            best_utility = float('-inf')
            for a in mdp.get_actions(state):
                exp_util = sum([ prob * utility[s_p] for s_p, prob in mdp.get_successor_probs(state, a).items() ])
                if exp_util > best_utility:
                    best_utility = exp_util
                    best_action = a
            policy[state] = best_action
    return policy


def ascii_grid_utils(utility):
    """Return an ascii-art gridworld with utilities.

    Args:
        utility: A dictionary mapping state (x, y) tuples to a utility value
    """
    return ascii_grid(dict([ (k, "{:8.4f}".format(v)) for k, v in utility.items() ]))


def ascii_grid_policy(actions):
    """Return an ascii-art gridworld with actions.

    Args:
        actions: A dictionary mapping state (x, y) tuples to an action (up, down, left, right)
    """
    symbols = { 'up':'^^^', 'right':'>>>', 'down':'vvv', 'left':'<<<', None:' x ' }
    return ascii_grid(dict([ (k, "   " + symbols[v] + "  ") for k, v in actions.items() ]))


def ascii_grid(vals):
    """helper function for printing out values associated with a 3x2 MDP."""
    state = ""
    state += " _____________________  \n"
    state += "|          |          | \n"
    state += "| {} | {} | \n".format(vals[(1, 3)], vals[(2, 3)])
    state += "|__________|__________| \n"
    state += "|          |          | \n"
    state += "| {} | {} | \n".format(vals[(1, 2)], vals[(2, 2)])
    state += "|__________|__________| \n"
    state += "|          |          | \n"
    state += "| {} | {} | \n".format(vals[(1, 1)], vals[(2, 1)])
    state += "|__________|__________| \n"
    return state


if __name__ == "__main__":
    
    forward_prob = 0.8
    def_reward = -1.0

    rows = 3
    cols = 2

    epsilon = 0.01
    gamma = 0.6
    
    rewards = {(1,3): -2, (2,3): 3}
    terminals = [(1,3), (2,3)]

    gridworld = MDP(rows, cols, rewards, terminals, forward_prob, def_reward)  

    utilities = value_iteration(gridworld, gamma, epsilon)  
    print(ascii_grid_utils(utilities))

    policy = derive_policy(gridworld, utilities)
    print(ascii_grid_policy(policy))
