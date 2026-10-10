"""Embed a 32-bit payload into an image and write a file ready to print."""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import codec
import embedder
import pattern


def paper_generator(amplitude, tile_path='tile22_fixed.npy'):
    """Their tile, phase-locked to the sync block, rotated by nearest neighbour."""
    tile = np.load(tile_path).astype(np.float64)
    th, tw = tile.shape

    def anchored(block_size, theta_deg, seed=None, interpolation=None,
                 rng_kind=None, dc=0.0, balanced=True, noise=0.0):
        diag = int(np.ceil(block_size * np.sqrt(2))) + 4
        offset = (diag - block_size) // 2
        ys, xs = np.mgrid[:diag, :diag]
        canvas = (tile[(ys - offset) % th, (xs - offset) % tw] + dc) * amplitude
        if abs(theta_deg) < 1e-9:                 # the sync block is not rotated
            return canvas[offset:offset + block_size, offset:offset + block_size]
        # Negated: cv2 turns the opposite way from the original generator.
        m = cv2.getRotationMatrix2D((diag / 2 - 0.5, diag / 2 - 0.5), -theta_deg, 1.0)
        rotated = cv2.warpAffine(canvas, m, (diag, diag), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_WRAP)
        return rotated[offset:offset + block_size, offset:offset + block_size]
    return anchored


MODES = {
    'plain':      dict(generator=None,   dc=0.0),
    'papercalib': dict(generator=0.67,   dc=1.28),
    'papermode':  dict(generator=1.0,    dc=0.65),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image', help='cover image; any size, it is resized to 512x512')
    ap.add_argument('--payload', help='8 hex digits (32 bits); random if omitted')
    ap.add_argument('--out', help='output prefix; defaults to the cover name')
    ap.add_argument('--mode', choices=sorted(MODES), default='plain')
    ap.add_argument('--tile', metavar='ФАЙЛ',
                    help='взять плитку из файла вместо псевдослучайной: .npy '
                         'с числами, либо картинка, где светлое даёт +1, '
                         'тёмное -1. Размер любой; поворот, сила и всё '
                         'остальное не меняются')
    ap.add_argument('--delta1', type=float, default=100.0)
    ap.add_argument('--delta2', type=float, default=10.0)
    ap.add_argument('--no-gamut-safe', dest='gamut_safe', action='store_false',
                    help='allow the pattern to saturate a colour channel; on by '
                         'default the strength is fitted to each pixel\'s own '
                         'colour headroom, which stops DC modes from flattening '
                         'the pattern into pale blobs in strongly coloured areas')
    args = ap.parse_args()

    if args.payload:
        payload = bytes.fromhex(args.payload)
        if len(payload) != codec.PAYLOAD_BYTES:
            sys.exit(f'payload must be {codec.PAYLOAD_BYTES} bytes '
                     f'({codec.PAYLOAD_BYTES * 2} hex digits)')
    else:
        payload = os.urandom(codec.PAYLOAD_BYTES)

    out = args.out or os.path.splitext(os.path.basename(args.image))[0] + '_wm'
    settings = MODES[args.mode]

    if args.tile:
        tile = pattern.load_tile(args.tile)
        pattern.set_tile(tile)
        print(f'плитка из {args.tile}: {tile.shape[0]}x{tile.shape[1]}, '
              f'значения {tile.min():g}..{tile.max():g}')

    original = pattern.make_pattern
    if settings['generator'] is not None:
        generator = paper_generator(settings['generator'])
        pattern.make_pattern = generator
        embedder.make_pattern = generator
    try:
        embedder.embed(args.image, f'{out}_core.png', f'{out}.json',
                       payload=payload, seed=1234,
                       delta1=args.delta1, delta2=args.delta2,
                       correct_jnd_kernels=True, jnd_beta_coef=0.01,
                       pattern_balanced=True, pattern_noise=0.0,
                       pattern_dc=settings['dc'], gamut_safe=args.gamut_safe)
        embedder.add_frame(f'{out}_core.png', f'{out}.png')
    finally:
        pattern.make_pattern = original
        embedder.make_pattern = original
        pattern.set_tile(None)

    print(f'\nполезная нагрузка: {payload.hex()}')
    print(f'режим {args.mode}, delta1={args.delta1:g} delta2={args.delta2:g}'
          f'{"" if args.gamut_safe else ", без учёта запаса цвета"}')
    print(f'печатать/показывать: {out}.png')


if __name__ == '__main__':
    main()
