from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamese.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN, NonLocalBAN


BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN,
        'NonLocalBAN': NonLocalBAN,
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)
