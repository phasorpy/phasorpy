"""Manage sample data files for testing and tutorials.

The ``phasorpy.datasets`` module provides a :py:func:`fetch` function to
download data files from remote repositories and cache them in a local
directory. The cache location can be changed by setting the
``PHASORPY_DATA_DIR`` environment variable.

Datasets from the following repositories are available:

- `PhasorPy tests <https://zenodo.org/record/8417894>`_
- `LFD Workshop <https://zenodo.org/record/8411056>`_
- `FLUTE <https://zenodo.org/record/8046636>`_
- `napari-flim-phasor-plotter
  <https://github.com/zoccoler/napari-flim-phasor-plotter/tree/0.0.6/src/napari_flim_phasor_plotter/data>`_
- `Phasor-based multi-harmonic unmixing for in-vivo hyperspectral imaging
  <https://zenodo.org/records/13625087>`_
  (`second record <https://zenodo.org/records/14860228>`_)
- `Convallaria slice acquired with time-resolved 2-photon microscope
  <https://zenodo.org/records/14026720>`_
- `Convallaria FLIM dataset in FLIM LABS JSON format
  <https://zenodo.org/records/15007900>`_

The implementation is based on the `Pooch <https://www.fatiando.org/pooch>`_
library.

"""

from __future__ import annotations

__all__ = ['fetch', 'REPOSITORIES']

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, Iterable

import pooch

ENV = 'PHASORPY_DATA_DIR'

DATA_ON_GITHUB = bool(
    os.environ.get('PHASORPY_DATA_ON_GITHUB', False)
) or bool(os.environ.get('GITHUB_ACTIONS', False))

TESTS = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/tests'
        if DATA_ON_GITHUB
        else 'doi:10.5281/zenodo.8417894'
    ),
    env=ENV,
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
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/lfd_workshop'
        if DATA_ON_GITHUB
        else 'doi:10.5281/zenodo.8411056'
    ),
    env=ENV,
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

FLUTE = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url='doi:10.5281/zenodo.8046636',
    env=ENV,
    registry={
        'Embryo.tif': (
            'sha256:'
            'd1107de8d0f3da476e90bcb80ddf40231df343ed9f28340c873cf858ca869e20'
        ),
        'Fluorescein_Embryo.tif': (
            'sha256:'
            '53cb66439a6e921aef1aa7f57ef542260c51cdb8fe56a643f80ea88fe2230bc8'
        ),
        'Fluorescein_hMSC.tif': (
            'sha256:'
            'a3f22076e8dc89b639f690146e46ff8a068388cbf381c2f3a9225cdcbbcec605'
        ),
        'hMSC control.tif': (
            'sha256:'
            '725570373ee51ee226560ec5ebb57708e2fac53effc94774c03b71c67a42c9f8'
        ),
        'hMSC-ZOOM.tif': (
            'sha256:'
            '6ff4be17e9d98a94b44ef13ec57af3c520f8deaeef72a7210ea371b84617ce92'
        ),
        'hMSC_rotenone.tif': (
            'sha256:'
            'cd0d2bd3baddc0f82c84c9624692e51bbbc56a80ac20b5936be0898d619c2bf2'
        ),
    },
)

NAPARI_FLIM_PHASOR_PLOTTER = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url='https://github.com/zoccoler/napari-flim-phasor-plotter/'
    'raw/0.0.6/src/napari_flim_phasor_plotter/data',
    env=ENV,
    registry={
        'hazelnut_FLIM_single_image.ptu': (
            'sha256:'
            '262f60ebc0054ba985fdda3032b58419aac07720e5f157800616c864d15fc2d3'
        ),
        'hazelnut_FLIM_z_stack.zip': (
            'sha256:'
            '8d26ebc7c758a70ee256d95c06f7921baa3cecbcdde82c7bb54b66bcb8db156e'
        ),
        'lifetime_cat.tif': (
            'sha256:'
            '5f2a2d20284a6f32fa3d1d13cb0c535cea5c2ec99c23148d9ee2d1e22d121a34'
        ),
        'lifetime_cat_labels.tif': (
            'sha256:'
            '102d74c202171f0ce2821dfbf1c92ead578bafebf99830e0cfa766e7407aadf9'
        ),
        'lifetime_cat_metadata.yml': (
            'sha256:'
            '20c447c1251598f255309fa866e58fc0e4abc2b73e824d18727833a05467d8bc'
        ),
        'seminal_receptacle_FLIM_single_image.sdt': (
            'sha256:'
            '2ba169495e533235cffcad953e76c7969286aad9181b946f5167390b8ff1a44a'
        ),
    },
)

ZENODO_13625087 = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/zenodo_13625087'
        # if DATA_ON_GITHUB
        # else 'doi:10.1088/2050-6120/ac9ae9'  # TODO: not working with Pooch
    ),
    env=ENV,
    registry={
        # part of ZENODO_14860228
        # '33_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm': (
        #    'sha256:'
        #    '68fcefcad4e750e9ec7068820e455258c986f6a9b724e66744a28bbbb689f986'
        # ),
        '34_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm': (
            'sha256:'
            '5c0b7d76c274fd64891fca2507013b7c8c9979d8131ce282fac55fd24fbb38bd'
        ),
        '35_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm': (
            'sha256:'
            'df57c178c185f6e271a66e2664dcc09d6f5abf923ee7d9c33add41bafc15214c'
        ),
        '38_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm': (
            'sha256:'
            '092ac050edf55e26dcda8cba10122408c6f1b81d19accf07214385d6eebfcf3e'
        ),
    },
)

ZENODO_14860228 = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/zenodo_14860228'
        if DATA_ON_GITHUB
        else 'doi:10.5281/zenodo.14860228'
    ),
    env=ENV,
    registry={
        '38_Hoechst_Golgi_Mito_Lyso_CellMAsk_404_488_561_633_SP.lsm': (
            'sha256:'
            '092ac050edf55e26dcda8cba10122408c6f1b81d19accf07214385d6eebfcf3e'
        ),
        'spectral cell mask.lsm': (
            'sha256:'
            'c4c2c567bd99ef4930d7278794d4e3daebaad0375c0852a5ab86a2ea056f4fe3'
        ),
        'spectral golgi.lsm': (
            'sha256:'
            'd0a5079d9ed18b1248434f3f6d4b2b240fb034891121262cfe9dfec64d8429cd'
        ),
        'spectral hoehst.lsm': (
            'sha256:'
            '3ee44a7f9f125698bb5e34746d9723669f67c520ffbf21244757d7fc25dbbb88'
        ),
        'spectral lyso tracker green.lsm': (
            'sha256:'
            '0964448649e2c73a57f5ca0c705c86511fb4625c0a2af0d7850dfa39698fcbb9'
        ),
        'spectral mito tracker.lsm': (
            'sha256:'
            '99b9892b247256ebf8a9917c662bc7bb66a8daf3b5db950fbbb191de0cd35b37'
        ),
    },
)

ZENODO_14976703 = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/zenodo_14976703'
        if DATA_ON_GITHUB
        else 'doi:10.5281/zenodo.14976703'
    ),
    env=ENV,
    registry={
        'Convalaria_LambdaScan.lif': (
            'sha256:'
            '27f1282cf02f87e11f8c7d3064066a4517ad4c9c769c796b32e459774f18c62a'
        ),
    },
)

CONVALLARIA_FBD = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/zenodo_14026720'
        if DATA_ON_GITHUB
        else 'doi:10.5281/zenodo.14026719'
    ),
    env=ENV,
    registry={
        'Convallaria_$EI0S.fbd': (
            'sha256:'
            '3751891b02e3095fedd53a09688d8a22ff2a0083544dd5c0726b9267d11df1bc'
        ),
        'Calibration_Rhodamine110_$EI0S.fbd': (
            'sha256:'
            'd745cbcdd4a10dbaed83ee9f1b150f0c7ddd313031e18233293582cdf10e4691'
        ),
    },
)

FLIMLABS = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/flimlabs'
        if DATA_ON_GITHUB
        else 'doi:10.5281/zenodo.15007900'
    ),
    env=ENV,
    registry={
        'Convallaria_m2_1740751781_phasor_ch1.json': (
            'sha256:'
            'a8bf0179f352ab2c6c78d0bd399545ab1fb6d5537f23dc06e5f12a7ef5af6615'
        ),
        'Convallaria_m2_1740751781_phasor_ch1.json.zip': (
            'sha256:'
            '9c5691f55e85778717ace13607d573bcd00c29e357e063c8db4f173340f72984'
        ),
        'Fluorescein_Calibration_m2_1740751189_imaging.json': (
            'sha256:'
            'aeebb074dbea6bff7578f409c7622b2f7f173bb23e5475d1436adedca7fc2eed'
        ),
        'Fluorescein_Calibration_m2_1740751189_imaging.json.zip': (
            'sha256:'
            '32960bc1dec85fd16ffc7dd74a3cd63041fb3b69054ee6582f913129b0847086'
        ),
        'Fluorescein_Calibration_m2_1740751189_imaging_calibration.json': (
            'sha256:'
            '7fd1b9749789bd139c132602a771a127ea0c76f403d1750e9636cd657cce017a'
        ),
    },
)

FIGSHARE_22336594 = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/figshare_22336594'
        if DATA_ON_GITHUB
        else 'doi:10.6084/m9.figshare.22336594.v1'
    ),
    env=ENV,
    registry={
        'FLIM_testdata.lif': (
            'sha256:'
            '902d8fa6cd39da7cf062b32d43aab518fa2a851eab72b4bd8b8eca1bad591850'
        ),
    },
)

FIGSHARE_22336594_EXPORTED = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url=(
        'https://github.com/phasorpy/phasorpy-data/raw/main/figshare_22336594'
    ),
    env=ENV,
    registry={
        'FLIM_testdata.lif.ptu': (
            'sha256:'
            'c85792b25d0b274f1484e490c99aa19052ab8e48e4e5022aabb1f218cd1123b6'
        ),
        'FLIM_testdata.lif.ptu.zip': (
            'sha256:'
            'c5134c470f6a3e5cb21eabd538cbd5061d9911dad96d58e3a4040cfddadaef33'
        ),
        'FLIM_testdata.xlef': (
            'sha256:'
            '7860ef0847dc9f5534895a9c11b979bb446f67382b577fe63fb166e281e5dc5e'
        ),
        'FLIM_testdata.xlef.zip': (
            'sha256:'
            'ad0ad6389f38dcba6f9809b54934ef3f19da975d9dabeb4c3a248692b959b9cf'
        ),
        'FLIM_testdata.lif.filtered.ptu': (
            'sha256:'
            'a00b84f626a39e79263322c60ae50b64163b36bb977ecc4dc54619097ba7f5b7'
        ),
        'FLIM_testdata.lif.filtered.ptu.zip': (
            'sha256:'
            '717366b231213bfd6e295c3efb7bf1bcd90e720eb28aab3223376087172e93e5'
        ),
    },
)

MISC = pooch.create(
    path=pooch.os_cache('phasorpy'),
    base_url='https://github.com/phasorpy/phasorpy-data/raw/main/misc',
    env=ENV,
    registry={
        'NADHandSHG.ifli': (
            'sha256:'
            'dfa65952850b8a222258776a8a14eb1ab7e70ff5f62b58aa2214797c5921b4a3'
        ),
    },
)

REPOSITORIES: dict[str, pooch.Pooch] = {
    'tests': TESTS,
    'lfd-workshop': LFD_WORKSHOP,
    'flute': FLUTE,
    'napari-flim-phasor-plotter': NAPARI_FLIM_PHASOR_PLOTTER,
    'zenodo-13625087': ZENODO_13625087,
    'zenodo-14860228': ZENODO_14860228,
    'zenodo-14976703': ZENODO_14976703,
    'convallaria-fbd': CONVALLARIA_FBD,
    'flimlabs': FLIMLABS,
    'figshare_22336594': FIGSHARE_22336594,
    'figshare_22336594_exported': FIGSHARE_22336594_EXPORTED,
    'misc': MISC,
}
"""Pooch repositories."""


def fetch(
    *args: str | Iterable[str | pooch.Pooch] | pooch.Pooch,
    extract_dir: str | None = '.',
    return_scalar: bool = True,
    **kwargs: Any,
) -> Any:  # str | tuple[str, ...]
    """Return absolute path(s) to sample file(s) in local storage.

    The files are downloaded from a remote repository if not present in local
    storage.

    Parameters
    ----------
    *args : str or iterable of str, optional
        Name(s) of file(s) or repositories to fetch from local storage.
        If omitted, return files in all repositories.
    extract_dir : str or None, optional
        Path, relative to cache location, where ZIP files will be unpacked.
    return_scalar : bool, optional
        If true (default), return single path as string, else tuple of string.
    **kwargs
        Additional arguments passed to ``pooch.fetch()``.
        For example, ``progressbar=True``.

    Returns
    -------
    str or tuple of str
        Absolute path(s) of file(s) in local storage.

    Examples
    --------
    >>> fetch('simfcs.r64')
    '...simfcs.r64'
    >>> fetch('simfcs.r64', 'simfcs.ref')
    ('...simfcs.r64', '...simfcs.ref')

    """
    filenames: list[str] = []
    if not args:
        args = tuple(REPOSITORIES.keys())
    for arg in args:
        if isinstance(arg, str):
            if arg in REPOSITORIES:
                # fetch all files in repository
                filenames.extend(
                    fetch(
                        *REPOSITORIES[arg].registry.keys(),
                        extract_dir=extract_dir,
                        return_scalar=False,
                        **kwargs,
                    )
                )
                continue
            for repo in REPOSITORIES.values():
                if arg + '.zip' in repo.registry:
                    # fetch single file in ZIP
                    filenames.append(
                        repo.fetch(
                            arg + '.zip',
                            processor=_Unzip([arg], extract_dir),
                            **kwargs,
                        )
                    )
                    break
                if arg in repo.registry:
                    if arg.endswith('.zip'):
                        # fetch and extract all files in ZIP
                        filenames.extend(
                            repo.fetch(
                                arg,
                                processor=pooch.processors.Unzip(
                                    extract_dir=extract_dir
                                ),
                                **kwargs,
                            )
                        )
                    else:
                        # fetch single file
                        filenames.append(repo.fetch(arg, **kwargs))
                    break
            else:
                raise ValueError(f'file {arg!r} not found')
        elif isinstance(arg, pooch.Pooch):
            # fetch all files in repository
            filenames.extend(
                fetch(
                    *arg.registry.keys(),
                    extract_dir=extract_dir,
                    return_scalar=False,
                    **kwargs,
                )
            )
        else:
            # fetch all files in iterable
            filenames.extend(
                fetch(
                    *arg,
                    extract_dir=extract_dir,
                    return_scalar=False,
                    **kwargs,
                )
            )
    if return_scalar and len(filenames) == 1:
        return filenames[0]
    return tuple(filenames)


class _Unzip(pooch.processors.ExtractorProcessor):  # type: ignore[misc]
    """Pooch processor that unpacks ZIP archive and returns single file."""

    def __call__(self, fname: str, action: str, pooch_: pooch.Pooch) -> str:
        pooch.processors.ExtractorProcessor.__call__(
            self, fname, action, pooch_
        )
        return os.path.splitext(fname)[0]

    @property
    def suffix(self) -> str:
        """String appended to unpacked archive folder name."""
        return '.unzip'

    def _all_members(self, fname: str) -> list[str]:
        """Return all members from archive."""
        from zipfile import ZipFile

        with ZipFile(fname, 'r') as zip_file:
            return zip_file.namelist()

    def _extract_file(self, fname: str, extract_dir: str) -> None:
        """Extract all files from ZIP archive."""
        from zipfile import ZipFile

        with ZipFile(fname, 'r') as zip_file:
            zip_file.extractall(path=extract_dir)
