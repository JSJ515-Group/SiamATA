from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamese.core.config import cfg
from siamese.tracker.siamese_tracker import SiameseTracker

TRACKS = {
          'SiameseTracker': SiameseTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
