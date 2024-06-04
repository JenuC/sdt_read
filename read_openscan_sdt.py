import sdtfile
import io
import zipfile
import numpy as np


def read_sdt_openscan(filename, print_on=False):
    """returns [data],t,[mblocks,bheads,info]"""

    with open(filename, "rb") as fid:

        ## HEADER
        header = np.rec.fromfile(
            fid, dtype=sdtfile.sdtfile.FILE_HEADER, shape=1, byteorder="<"
        )[0]
        # print(header)
        if print_on:
            FH = sdtfile.sdtfile.FILE_HEADER
            FH = [i[0] for i in FH]
            for j in zip(FH, header):
                print(j)

        ## INFO

        if "info_offset" in header.dtype.names:
            fid.seek(header.info_offset)
        else:
            fid.seek(header.info_offs)
        info = fid.read(header.info_length).decode("windows-1250")
        info = info.replace("\r\n", "\n")
        if print_on:
            print(info)

        ## SETUP
        fid.seek(header.setup_offs)
        fsetup = sdtfile.SetupBlock(fid.read(header.setup_length))  # cant read this !

        ## MEAS_BLOCK
        if "meas_desc_block_offset" in header.dtype.names:
            fh = fid.seek(header.meas_desc_block_offset)
        else:
            fh = fid.seek(header.meas_desc_block_offs)

        mblocks = []
        for measblock_index in range(header.no_of_meas_desc_blocks):
            # fmeas_blk = fid.read()
            mblocks.append(
                np.rec.fromfile(
                    fid,
                    dtype=np.dtype(sdtfile.sdtfile.MEASURE_INFO),
                    shape=1,
                    byteorder="<",
                )
            )
            fid.seek(
                header.meas_desc_block_length
                - np.dtype(sdtfile.sdtfile.MEASURE_INFO).itemsize,
                1,
            )

        if print_on:
            # MH = sdtfile.sdtfile.MEASURE_INFO
            # MH= [i[0] for i in MH]
            for j in zip(mblocks[0].dtype.names, mblocks[0][0]):
                print(j)

        # assuming sinilar channels
        adc_re = int(mblocks[0].adc_re)
        image_x = int(mblocks[0].image_x)
        image_y = int(mblocks[0].image_y)
        t = np.arange(adc_re, dtype="float64")
        t *= mblocks[0].tac_r / float(mblocks[0].tac_g * adc_re)

        ## BLOCK HEADER AND DATABLOCKS :
        bheads = []
        data_list = []

        ## MEAS_BLOCK
        if "data_block_offset" in header.dtype.names:
            offset = header.data_block_offset
        else:
            offset = header.data_block_offs

        for datablock_index in range(header.no_of_data_blocks):

            fid.seek(offset)
            if hasattr(sdtfile.sdtfile, "BLOCK_HEADER_15"):
                bh = np.rec.fromfile(
                    fid, dtype=sdtfile.sdtfile.BLOCK_HEADER_15, shape=1, byteorder="<"
                )[0]
            else:
                bh = np.rec.fromfile(
                    fid, dtype=sdtfile.sdtfile.BLOCK_HEADER_OLD, shape=1, byteorder="<"
                )[0]
            bheads.append(bh)
            fid.seek(bh.data_offs)

            bt = sdtfile.sdtfile.BlockType(bh.block_type)
            dtype = bt.dtype
            dsize = bh.block_length // dtype.itemsize

            if print_on:
                print(datablock_index)
                for j in zip(bh.dtype.names, bh):
                    print(j)

            if sdtfile.sdtfile.BlockType(bh.block_type).compress:
                bio = io.BytesIO(fid.read(bh.next_block_offs - bh.data_offs))
                with zipfile.ZipFile(bio) as zf:
                    data = zf.read("data_block")
                del bio
                data = np.frombuffer(data, dtype=dtype, count=dsize)
            else:
                data = np.fromfile(fid, dtype=dtype, count=dsize)

            data = data.reshape(image_x, image_y, adc_re)
            data_list.append(data)

            offset = bh.next_block_offs
        return (data_list, t, [mblocks, bheads, info])
