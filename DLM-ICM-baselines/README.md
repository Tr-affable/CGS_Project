The above code generates random baseline trees that match the natural language trees in the number of nodes, number of crossings, and distribution of dependency lengths and intervener complexity. See <a href="https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00060/112598/A-Reappraisal-of-Dependency-Length-Minimization-as">this paper</a> for more details. 

Each module with the name "consruct_output... .py" takes a directory containing conllu format files, and computes the formal measures for real trees and corresponding random baseline trees. There are six such modules for six baselines. The measures being computed i.e., gap degree, edge degree etc. are stored in same output file for real trees and random baseline trees.

The modules with name "baseline_conditions... .py" containts the algorithm to generate trees for different baselines. To know more about these baselines and measures being computed, see <a href="https://sites.socsci.uci.edu/~rfutrell/papers/yadav2021dependency.pdf"> this paper</a>.

To install required libraries, run: <code>pip install -r requirements.txt</code>
