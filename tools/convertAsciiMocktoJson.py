# ================================================================================================
# 
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License, version 2.1, as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
#
# ================================================================================================
import os
import sys
import json


def read_ascii_mock_file(ascii_mock_filename):
    meta_data_dict = dict()
    with open(ascii_mock_filename, 'r') as ascii_file:
        ascii_text = ascii_file.readlines()

    # line 0:

    # //f << "rawDataMockMetadata v 1";
    # std::string dummy;
    # int version;
    # f >> dummy;
    # f >> dummy;
    # f >> version;
    assert(ascii_text[0].strip() == "rawDataMockMetadata v 3"), "ascii mock file might have unsupported format"

    # line 1:
    line = ascii_text[1].strip().split()
    line.reverse()

    # size_t numElements;
    # f >> numElements;
    meta_data_dict["numElements"] = int(line.pop())

    # vec2s elementLayout;
    # f >> elementLayout.x;
    # f >> elementLayout.y;
    meta_data_dict["elementLayout"] = {
        'x': int(line.pop()),
        'y': int(line.pop())
    }

    # size_t numReceivedChannels;
    # f >> numReceivedChannels;
    meta_data_dict["numReceivedChannels"] = int(line.pop())

    # size_t numSamples;
    # f >> numSamples;
    meta_data_dict["numSamples"] = int(line.pop())

    # size_t numTxScanlines;
    # f >> numTxScanlines;
    meta_data_dict["numTxScanlines"] = int(line.pop())

    # vec2s scanlineLayout;
    # f >> scanlineLayout.x;
    # f >> scanlineLayout.y;
    meta_data_dict["scanlineLayout"] = {
        'x': int(line.pop()),
        'y': int(line.pop())
    }

    # double depth;
    # f >> depth;
    meta_data_dict["depth"] = float(line.pop())

    # double samplingFrequency;
    # f >> samplingFrequency;
    meta_data_dict["samplingFrequency"] = float(line.pop())

    # size_t rxNumDepths;
    # f >> rxNumDepths;
    meta_data_dict["rxNumDepths"] = int(line.pop())

    # double speedOfSoundMMperS;
    # f >> speedOfSoundMMperS;
    meta_data_dict["speedOfSoundMMperS"] = float(line.pop())

    # line 2:
    #
    line = ascii_text[2].strip().split()
    line.reverse()

    # size_t numRxScanlines = scanlineLayout.x*scanlineLayout.y;
    #
    # std::vector<ScanlineRxParameters3D> rxScanlines(numRxScanlines);
    #
    # for (size_t scanlineIdx = 0; scanlineIdx < numRxScanlines; idx++)
    # {
    #     ScanlineRxParameters3D params;
    #     f >> params;
    #     rxScanlines[scanlineIdx] = params;
    # }
    num_rx_scanlines = meta_data_dict["scanlineLayout"]['x'] * meta_data_dict["scanlineLayout"]['y']
    rx_scanlines = []
    for k in range(num_rx_scanlines):
        params = dict()
        params['position'] = {
            'x': float(line.pop()),
            'y': float(line.pop()),
            'z': float(line.pop())
        }
        params['direction'] = {
            'x': float(line.pop()),
            'y': float(line.pop()),
            'z': float(line.pop())
        }
        params['maxElementDistance'] = {
            'x': float(line.pop()),
            'y': float(line.pop())
        }

        txParameters = []
        for m in range(4):
            txParams = {
                'firstActiveElementIndex': {
                    'x': int(line.pop()),
                    'y': int(line.pop())
                },
                'lastActiveElementIndex': {
                    'x': int(line.pop()),
                    'y': int(line.pop())
                },
                'txScanlineIdx': int(line.pop()),
                'initialDelay': float(line.pop()),
                'txWeights': float(line.pop())
            }
            txParameters.append(txParams)
        params['txParameters'] = txParameters
        rx_scanlines.append(params)

    meta_data_dict['rxScanlines'] = rx_scanlines

    # line 3:
    #
    line = ascii_text[3].strip().split()
    line.reverse()
    # for (size_t idx = 0; idx < rxNumDepths; idx++)
    # {
    #     LocationType val;
    #     f >> val;
    #     rxDepths[idx] = val;
    # }
    # This will not be needed in the new format!


    # line 4:
    #
    line = ascii_text[4].strip().split()
    line.reverse()

    meta_data_dict['rxElementPosition'] = []
    # for (size_t idx = 0; idx < numElements; idx++)
    # {
    #     LocationType val;
    #     f >> val;
    #     rxElementXs[idx] = val;
    # }
    for k in range(meta_data_dict['numElements']):
        meta_data_dict['rxElementPosition'].append({'x': float(line.pop())})

    # for (size_t idx = 0; idx < numElements; idx++)
    # {
    #     LocationType val;
    #     f >> val;
    #     rxElementYs[idx] = val;
    # }
    for k in range(meta_data_dict['numElements']):
        meta_data_dict['rxElementPosition'][k]['y'] = float(line.pop())

    return meta_data_dict


def write_json_mock_file(meta_data_dict, json_mock_filename):
    with open(json_mock_filename, 'w') as outfile:
        json.dump(meta_data_dict, outfile, indent=2)
    pass


def print_usage():
    print("Usage: {} <ascii-mock-filename source> <json-mock-filename out>".format(sys.argv[0]))


def main(argv):
    if len(argv) != 2:
        print_usage()
        return

    meta_data_dict = read_ascii_mock_file(argv[0])
    write_json_mock_file(meta_data_dict, argv[1])
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
