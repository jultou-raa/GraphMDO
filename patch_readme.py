import re

with open("README.md", "r") as f:
    text = f.read()

# Replace mentions of ax-platform for loop execution or SMT with gemseo.
# Although ax-platform is still used as the algorithm, we now use gemseo's DOEs and Surrogates!
text = re.sub(r'The optimization service uses `ax-platform` for Bayesian Optimization and `smt` for surrogate modeling.',
              r'The optimization service delegates execution strictly to `gemseo` native `MDOScenario` and `DOEScenario` features. Bayesian Optimization is orchestrated via a custom algorithm wrapper delegating to `ax-platform`, while surrogate models utilize `gemseo` native Machine Learning regression plugins.', text)

with open("README.md", "w") as f:
    f.write(text)
