# iopsy
A python library for industrial-organizational psychology

This is a work in progress, but in a dream state would cover anything somebody doing I-O work could want. Current coverage of the domain is limited to:
- Adverse Impact
  - core is complete, timely given the new New York law regarding annual adverse impact testing
  - reporting options would be nice to have
- Analysis
  - base class and a bunch of filtering options are done
  - need to decide on a few common use case analyses to form up into specialized analysis classes inheriting from the base
  - meta analysis class to interact with other analysis classes
- Job Analysis
  - Havent started this but would like to have some tools here to process job analyses more easily
  - Automatic work and worker linkage mapping
  - Automatic role clustering
  - Visualizations
  - Performance appraisal generation
- Psychometrics
  - Got decent coverage of CTT up as well as some convenience functions
  - Really would like to get some IRT going, but thats a lot of math to think through. Need to find a good resource beyond embretson and reise.
  - Would like to get some NLP based psychometrics going as well, measuring item sentiment, looking at topic models of scales, look at shared dictionaries across scales for potential overlap, semantic similarity to the definition or described domain space.
  - Want to throw together a item, scale, and psych data class to make working with psych data easier. item would have things like IRT parameters, and ICC attributes. Scales would be compositions of items, offering scoring options and easy access to the items of which it is composed, lots of scale statistics and aggregations of item statistics (alpha, factor loadings, CITrs, etc.). psych data classes will be a culmination of scales, offering access to the item level as well, but also offering multi scale methods, like item other correlations, and loadings across scales
- Selection
  - Currently a single algorithm, a weighted regularized regression allowing for diminishing weight placed on variables that show group differences as a tunable parameter.
  - So much more to do here, lots of fun algos to try out (l2 weighted regularization is an obvious next step)
  - build out model explainability metrics, maybe an implementation of RWA and dominance analysis
