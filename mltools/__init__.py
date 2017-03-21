
# import needed utility functions into main namespace
from .utils import *

# import "base" (abstract) classifier and regressor classes
from .base import *

# import feature transforms etc into sub-namespace
from . import transforms 
from .transforms import rescale        # useful to have at top namespace

# import "plot" functions into main namespace
from .plot import *

# import classifiers into sub-namespaces
try: from . import bayes
except ImportError: pass

try: from . import knn
except ImportError: pass

try: from . import linear
except ImportError: pass

try: from . import linearC
except ImportError: pass

try: from . import nnet
except ImportError: pass

try: from . import dtree
except ImportError: pass

try: from . import ensembles
except ImportError: pass

# import clustering unsupervised learning algos
try: from . import cluster
except ImportError: pass


