"""Manage sample data files for testing and tutorials.

The ``phasorpy.datasets`` module provides a :py:func:`fetch` function to
download data files from remote repositories.
The downloaded files are cached in a local directory.

The implementation is based on the `Pooch <https://www.fatiando.org/pooch>`_
library.

"""

from __future__ import annotations

__all__ = ['fetch']

from typing import Any

import pooch

TESTS = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url='doi:10.5281/zenodo.8417894',
    registry={
        'flimage.int.bin': (
            'sha256:'
            '5d470ed31ed0611b43270261341bc1c41f55fda665eaf529d848a139fcae5fc8'
        ),
        'flimage.int.bin.zip': (
            'sha256:'
            '51062322891b4c22d577100395d8c02297c5494030d2c550a0fd6c90f73cc211'
        ),
        'flimage.mod.bin': (
            'sha256:'
            'b0312f6f9f1f24e228b2c3a3cb07e18d80b35f31109f8c771b088b54419c5200'
        ),
        'flimage.phi.bin': (
            'sha256:'
            'd593acc698a1226c7a4571fa61f6751128584dcca6ed4449d655283cd231b125'
        ),
        'flimfast.flif': (
            'sha256:'
            'b12cedf299831a46baf80dcfe7bfb9f366fee30fb3e2b3039d4346f1bbaf3e2c'
        ),
        'flimfast.flif.zip': (
            'sha256:'
            'b25642b1a2dbc547f0cdaadc13ce89ecaf372f90a20978910c02b13beb138c2e'
        ),
        'frequency_domain.ifli': (
            'sha256:'
            '56015b98a2edaf4ee1262b5e1034305aa29dd8b20e301ced9cd7a109783cd171'
        ),
        'frequency_domain.ifli.zip': (
            'sha256:'
            '93066cc48028360582f6e3380d09d2c5a6f540a8f931639da3cfca158926df9b'
        ),
        'paramecium.lsm': (
            'sha256:'
            'b3b3b80be244a41352c56390191a50e4010d52e5ca341dc51bd1d7c89f10cedf'
        ),
        'paramecium.lsm.zip': (
            'sha256:'
            '7828a80e878ee7ab88f9bd9a6cda72e5d698394d37f69a7bee5b0b31b3856919'
        ),
        'simfcs.b64': (
            'sha256:'
            '5ccccd0bcd46c66ea174b6074975f631bdf163fcb047e35f9310aaf67c320fb8'
        ),
        'simfcs.b64.zip': (
            'sha256:'
            'b176761905fc5916d0770bd6baaa0e31af5981f92ec323e588f9ce398324818e'
        ),
        'simfcs.b&h': (
            'sha256:'
            '6a406c4fd862318a51370461c7607390f022afdcb2ce27c4600df4b5af83c26e'
        ),
        'simfcs.b&h.zip': (
            'sha256:'
            'ec14666be76bd5bf2dcaee63435332857795173c8dc94be8778697f32b041aa1'
        ),
        'simfcs.bhz': (
            'sha256:'
            '14f8b5287e257514945ca17a8398425fc040c00bfad2d8a3c6adb4790862d211'
        ),
        'simfcs.r64': (
            'sha256:'
            'ead3b2df45c1dff91e91a325f97225f4837c372db04c2e49437ee9ec68532946'
        ),
        'simfcs.ref': (
            'sha256:'
            '697dad17fb3a3cf7329a45b43ba9ae5f7220c1f3d5f08749f2ce3eadb0598420'
        ),
        'simfcs.ref.zip': (
            'sha256:'
            '482331ec5586973e9eb5c2e4f793d9621cc756ce8f4093f5e21906a7ce5726f8'
        ),
        'simfcs.z64': (
            'sha256:'
            'f1dd861f80528c77bb581023c05c7bf7293d6ad3c4a3d10f9a50b8b5618a8099'
        ),
        'simfcs_1000.int': (
            'sha256:'
            'bb5bde0ecf24243865cdbc2b065358fe8c557696de18567dbb3f75adfb2ab51a'
        ),
        'simfcs_1000.int.zip': (
            'sha256:'
            'f75e211dc344f194a871f0b3af3f6d8a9e4850e9718526f2bfad87ef16c1c377'
        ),
        'simfcs_1000.mod': (
            'sha256:'
            '84265df08d48ff56f6844d55392fccac9fa429c481d1ac81b07c23738075d336'
        ),
        'simfcs_1000.phs': (
            'sha256:'
            '8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90'
        ),
        'tcspc.sdt': (
            'sha256:'
            '0ff0b25b36cb9a7657112a3b081ff479bcae487ce8100b8756e5780e0957708d'
        ),
        'tcspc.sdt.zip': (
            'sha256:'
            '57a772bc413e85e0f13eb996f8c2484dfb3d15df67ffa6c3b968d3a03c27fdc3'
        ),
    },
)

LFD_WORKSHOP = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url='doi:10.5281/zenodo.8411056',
    registry={
        '4-22-03-2-A5-CHO-CELL3B.tif': (
            'sha256:'
            '015d4b5a4cbb6cc40ac0c39f7a0b57ff31173df5b3112627b551e4d8ce8c3b02'
        ),
        '1011rac1002.ref': (
            'sha256:'
            'b8eb374d21ba74519342187aa0b6f67727983c1e9d02a9b86bde7e323f5545ac'
        ),
        'CFP and CFP-YFp.ref': (
            'sha256:'
            'f4f494d5e71836aeacfa8796dcf9b92bbc0f62b8176d6c10d5ab9ce202313257'
        ),
        'CFP-YFP many cells with background.ref': (
            'sha256:'
            '7cb88018be807144edbb2746d0ec6548eeb3ddc4aa176f3fba4223990aa21754'
        ),
        'CFPpax8651866.ref': (
            'sha256:'
            'eda9177f2841229d120782862779e2db295ad773910b308bc2c360c22c75f391'
        ),
        'Paxillins013.bin': (
            'sha256:'
            'b979e3112dda2fa1cc34a351dab7b0c82009ef07eaa34829503fb79f7a6bb7d2'
        ),
        'capillaries1001.ref': (
            'sha256:'
            '27f071ae31032ed4e79c365bb2076044876f7fc10ef622aff458945f33e60984'
        ),
        'pax1023.bin': (
            'sha256:'
            'f467e8264bb10fc12a19506693837f384e32ca01c0cac0b25704c19ceb8d7d5a'
        ),
    },
)

REPOSITORIES: tuple[pooch.Pooch, ...] = (TESTS, LFD_WORKSHOP)
"""Pooch repositories."""


def fetch(
    filename: str,
    /,
    *,
    extract_dir: str | None = '.',
    **kwargs: Any,
) -> str:
    """Return absolute path to sample file in local storage.

    The file is downloaded from a remote repository if the file does not
    already exist in the local storage.

    Parameters
    ----------
    filename : str
        Name of file to fetch from local storage.
    extract_dir : str or None, optional
        Path, relative to cache location, where ZIP files will be unpacked.
    **kwargs : optional
        Additional parameters passed to ``pooch.fetch()``.

    Returns
    -------
    full_path : str
        Absolute path of file in local storage.

    Examples
    --------
    >>> fetch('simfcs.r64')
    '...simfcs.r64'

    """
    for repo in REPOSITORIES:
        if filename + '.zip' in repo.registry:
            # download and extract ZIP, return file name
            return repo.fetch(
                filename + '.zip',
                processor=_Unzip([filename], extract_dir),
                **kwargs,
            )
        if filename in repo.registry:
            if filename.endswith('.zip'):
                # download and extract ZIP, return all file names in ZIP
                return repo.fetch(
                    filename,
                    processor=pooch.processors.Unzip(extract_dir=extract_dir),
                    **kwargs,
                )
            # download file, return file name
            return repo.fetch(filename, **kwargs)
    raise ValueError('file not found')


class _Unzip(pooch.processors.ExtractorProcessor):
    """Pooch processor that unpacks ZIP archive and returns single file."""

    def __call__(self, fname, action, pooch_):
        filenames = pooch.processors.ExtractorProcessor.__call__(
            self, fname, action, pooch_
        )
        return filenames[0]

    def _extract_file(self, fname, extract_dir):
        """Extract all files from ZIP archive."""
        from zipfile import ZipFile

        with ZipFile(fname, 'r') as zip_file:
            zip_file.extractall(path=extract_dir)
