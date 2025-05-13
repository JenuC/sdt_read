import sdtfile
import numpy as np
import zipfile
import pathlib
import io
from sdtfile import SdtFile


def read_sdt_info_brukerSDT(filename):
    """
    modified from CGohlke sdtfile.py to read bruker 150 card data
    gives tarr, x.shape,y.shape,t.shape,c.shape
    """
    ## HEADER
    with open(filename, "rb") as fh:
        header = np.rec.fromfile(
            fh, dtype=sdtfile.sdtfile.FILE_HEADER, shape=1, byteorder="<"
        )

    measure_info = []
    #dtype = np.dtype(sdtfile.sdtfile.MEASURE_INFO)
    ## meeting requirements from https://github.com/cgohlke/sdtfile/commit/4c574b437aa8cefefaf700d804ec66ae11a6c22c
    dtype = sdtfile.sdtfile.record_dtype(
            sdtfile.sdtfile.MEASURE_INFO, int(header.meas_desc_block_length)
        )
    #print(dtype)
    with open(filename, "rb") as fh:
        if "meas_desc_block_offset" in header.dtype.names:
            fh.seek(header.meas_desc_block_offset[0])
        elif "meas_desc_block_offs" in header.dtype.names:
            fh.seek(header.meas_desc_block_offs[0])
        else:
            raise ValueError(
                "Header does not contain meas_desc_block_offset or meas_desc_block_offs"
            )
        for _ in range(header.no_of_meas_desc_blocks[0]):
            measure_info.append(
                np.rec.fromfile(fh, dtype=dtype, shape=1, byteorder="<")
            )
            fh.seek(header.meas_desc_block_length[0] - dtype.itemsize, 1)

    times = []
    block_headers = []

    if "data_block_offset" in header.dtype.names:
        offset = header.data_block_offset[0]
    elif "data_block_offs" in header.dtype.names:
        offset = header.data_block_offs[0]
    else:
        raise ValueError("Header does not contain data_block_offset or data_block_offs")
    # print()
    with open(filename, "rb") as fh:
        for _ in range(header.no_of_data_blocks[0]):  ##
            fh.seek(offset)
            # read data block header
            bh = np.rec.fromfile(
                fh, dtype=sdtfile.sdtfile.BLOCK_HEADER, shape=1, byteorder="<"
            )[0]
            block_headers.append(bh)
            # read data block
            mi = measure_info[bh.meas_desc_block_no]

            dtype = sdtfile.sdtfile.BlockType(bh.block_type).dtype
            dsize = bh.block_length // dtype.itemsize

            # t = np.arange(mi.adc_re[0], dtype=np.float64) # numpy after 1.25 reads them as array with ndim=1
            # t *= mi.tac_r / float(mi.tac_g * mi.adc_re)
            t = np.arange(mi["adc_re"][0], dtype=np.float64)
            t *= mi["tac_r"] / mi["tac_g"] * mi["adc_re"]

            if "image_rx" in mi.dtype.names:
                routing_channels_x = mi["image_rx"][0]
            else:
                routing_channels_x = 1

            times.append(t)
            offset = bh.next_block_offs
        
        ## SCAN-SYNC-IN format loves the imagesizes in scan_x,scan_y instead of image_x,image_y
        ## need to check large file- adc.re>512 that breaks the code
        axis_dim = [
            int(x)
            for x in [mi.scan_x[0], mi.scan_y[0], mi.adc_re[0], routing_channels_x]
        ]

        return (times, axis_dim)


def read_sdt150(filename):
    """sdt bruker uses data_block001 instead of data_block"""
    # TODO: Need to move into simpler sdtfile-py reading for scan-sync in support
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    t, XYTC = read_sdt_info_brukerSDT(filename)
    x, y, t, c = [int(a) for a in XYTC]
    try:
        with zipfile.ZipFile(filename) as myzip:
            infolist = myzip.infolist()
            if not infolist:
                raise ValueError("Zip file is empty.")
            z1 = infolist[0]  # e.g., "data_block"
            with myzip.open(z1.filename) as myfile:
                dataspl = myfile.read()
    except zipfile.BadZipFile:
        # print(f"Not a valid zip file: {filename}")

        with open(filename, "rb") as f:
            dataspl = f.read()
            dataspl = dataspl[len(dataspl) - (x * y * t * c * 2) :]
    except Exception as e:
        print(f"An error occurred: {e}")
        dataspl = None

    dataSDT = np.fromstring(dataspl, np.uint16)

    if c > 1:
        dataSDT = dataSDT[: x * y * t * c].reshape([c, x, y, t])
        if (
            dataSDT[0, :, :, :].sum() == 0
        ):  # bruker uses two channels and keeps one empty!!!
            dataSDT = np.squeeze(dataSDT[1, :, :, :])
        else:
            if (
                dataSDT[1, :, :, :].sum() == 0
            ):  # bruker uses two channels and keeps one empty!!!
                dataSDT = np.squeeze(dataSDT[0, :, :, :])
                # print ' ch1 is empty'
            else:
                pass
    else:
        dataSDT = dataSDT[: x * y * t * c].reshape([x, y, t])
    # print("READ DATA IN:",dataSDT.shape)
    return dataSDT


def read_bruker_sdt(filename):
    return read_sdt150(filename)
    
