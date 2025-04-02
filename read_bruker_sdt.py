import sdtfile
import numpy as np
import zipfile


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
    dtype = np.dtype(sdtfile.sdtfile.MEASURE_INFO)
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
        return (times, [mi.scan_x[0], mi.scan_y[0], mi.adc_re[0], routing_channels_x])


def read_sdt150(filename):
    """sdt bruker uses data_block001 instead of data_block"""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    t, XYTC = read_sdt_info_brukerSDT(filename)

    with zipfile.ZipFile(filename) as myzip:
        z1 = myzip.infolist()[0]  # "data_block"
        with myzip.open(z1.filename) as myfile:
            dataspl = myfile.read()

    dataSDT = np.fromstring(dataspl, np.uint16)

    if XYTC[3] > 1:
        dataSDT = dataSDT[: XYTC[0] * XYTC[1] * XYTC[2] * XYTC[3]].reshape(
            [XYTC[3], XYTC[0], XYTC[1], XYTC[2]]
        )
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
        dataSDT = dataSDT[: XYTC[0] * XYTC[1] * XYTC[2] * XYTC[3]].reshape(
            [XYTC[0], XYTC[1], XYTC[2]]
        )
    # print("READ DATA IN:",dataSDT.shape)
    return dataSDT


def read_bruker_sdt(filename):
    """
    Reads a Bruker .sdt file and returns a list of reshaped data blocks.

    Parameters:
        filename (str): Path to the .sdt file.

    Returns:
        list of np.ndarray: Reshaped data blocks with shape (channels, x, y, time).
    """
    sdt = sdtfile.SdtFile(filename)
    data_blocks = []

    for index, block in enumerate(sdt.data):
        scan_info = sdt.measure_info[index]
        x, y, t = scan_info.scan_x, scan_info.scan_y, scan_info.adc_re
        channels = scan_info.image_rx

        if block.size == x * y * t * channels:
            block = block.reshape((channels, x, y, t))

        data_blocks.append(block.copy())

    return data_blocks
