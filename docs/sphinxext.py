# Configuration file for Sphinx tutorial order.

class TutorialOrder:
    """Order tutorials in gallery subsections."""

    tutorials = [
        'introduction',
        'lifetime_geometry',
        'lfd_workshop',
        # api
        'io',
        'phasor_from_lifetime',
        'multi-harmonic',
        'filtering',
        'phasorplot',
        'cursors',
        'components',
        'fret',
        'lifetime_to_signal',
        'pca',
        # benchmarks
        'phasor_from_signal',
    ]

    def __init__(self, srcdir: str): ...

    def __call__(self, filename: str) -> int:
        return self.tutorials.index(filename[9:-3])
