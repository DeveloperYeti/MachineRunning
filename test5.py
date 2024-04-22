import numpy as np
patoffs = np.array([1.0,-0.5])
probs = np.array([0.5,0.5])

def is_even(n):
    return n % 2 == 0
winnings = 0.0
for toss_ct in range(10):
    die_toss = np.random.randint(1,7)
    winnings +=1.0 if is_even(die_toss) else -0.5
print(winnings)