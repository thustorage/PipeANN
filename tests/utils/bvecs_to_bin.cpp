// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"
#include "logger.h"

void block_convert(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf, _u64 npts,
                   _u64 ndims) {
  reader.read((char *) read_buf, npts * (ndims * sizeof(uint8_t) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + sizeof(unsigned))) + sizeof(unsigned),
           ndims * sizeof(uint8_t));
  }
  writer.write((char *) write_buf, npts * ndims * sizeof(uint8_t));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    diskann::cout << argv[0] << " input_bvecs output_bin" << std::endl;
    exit(-1);
  }
  std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
  _u64 fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);

  unsigned ndims_u32;
  reader.read((char *) &ndims_u32, sizeof(unsigned));
  reader.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = fsize / (ndims + sizeof(unsigned));
  diskann::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  diskann::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[2], std::ios::binary);
  int npts_s32 = (_s32) npts;
  int ndims_s32 = (_s32) ndims;
  writer.write((char *) &npts_s32, sizeof(_s32));
  writer.write((char *) &ndims_s32, sizeof(_s32));
  uint8_t *read_buf = new uint8_t[npts * (ndims + sizeof(unsigned))];
  uint8_t *write_buf = new uint8_t[npts * ndims];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
    diskann::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;
  delete[] write_buf;

  reader.close();
  writer.close();
}
