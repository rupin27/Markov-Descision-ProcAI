# Markov-Descision-ProcAI
# Description
<div>
The program is a modularized code that has been implemented using Python. The goal was to implement Value Iteration for grid world Markov Descision Problems and observe the effects of different parameters on policies.
<div>  
The gridworld MDP is shown below. 
</div>
  
<div>
  
<pre>
<img src="MDP/mdp.png" height="315" width="350">           <img src="MDP/movement.png" height="150" width="170"> 

The single terminal state (1, 3) has a reward of +10, the non-terminal (1, 2) has reward -5, and all other states 
have a reward of -1. 
The agent makes its intended move (up, down, left, or right) with a probability 0.8, and moves in a perpendicular 
direction with probability 0.1 for each side (e.g., if intending to go right, the agent can move up or down with 
a probability of 0.1 each). If the agent runs into a wall, it stays in the same place.
</pre>
<div>
The goal in a Markov decision process is to find a good "policy" for the decision maker: a function π that specifies the action π(s) that the decision maker will choose when in state s. Once a Markov decision process is combined with a policy in this way, this fixes the action for each state and the resulting combination behaves like a Markov chain (since the action chosen in state s is completely determined by π(s) and <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}Pr(s_{t&plus;1}&space;=&space;s'&space;|&space;s_{t}&space;=&space;s,&space;a_{t}&space;=&space;t&space;)&space;" title="https://latex.codecogs.com/png.image?\inline \dpi{110}\bg{white}Pr(s_{t+1} = s' | s_{t} = s, a_{t} = t ) " /> reduces to <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}Pr(s_{t&plus;1}&space;=&space;s'&space;|&space;s_{t}&space;=&space;s)&space;" title="https://latex.codecogs.com/png.image?\inline \dpi{110}\bg{white}Pr(s_{t+1} = s' | s_{t} = s) " /> a Markov transition matrix).

The objective is to choose a policy π that will maximize some cumulative function of the random rewards, typically the expected discounted sum over a potentially infinite horizon:

<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}E[\sum_{t&space;=&space;0}^{\infty&space;}\gamma&space;^{t}R_{a_{t}}(s_{t},&space;s)]&space;" title="https://latex.codecogs.com/png.image?\inline \dpi{110}\bg{white}E[\sum_{t = 0}^{\infty }\gamma ^{t}R_{a_{t}}(s_{t}, s)] " /> (where we choose <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}a(t)&space;=&space;\pi&space;(s_{t})&space;" title="https://latex.codecogs.com/png.image?\inline \dpi{110}\bg{white}a(t) = \pi (s_{t}) " /> i.e. actions given by the policy). And the expectation is taken over <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;s_{t&plus;1}\sim&space;P_{a_{t}}(s_{t},&space;s_{t&plus;1})" title="https://latex.codecogs.com/png.image?\inline \dpi{110}\bg{white} s_{t+1}\sim P_{a_{t}}(s_{t}, s_{t+1})" /> where γ is the discount factor satisfying 0 ≤ γ ≤ 1, which is usually close to 1 (for example, γ = 1/(1+r) for some discount rate r). A lower discount factor motivates the decision maker to favor taking actions early, rather than postpone them indefinitely.

</div> 
  

