"""Read-only local viewer for synchronized AV/UL/UR/LL/LR landmarks."""

import argparse
import base64
import errno
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import subprocess
from urllib.parse import parse_qs, urlparse

import cv2
import h5py
import numpy as np

from .schema import ACTORS, HANDEDNESS, sha256, validate_hdf5
from .locations import Locations, contained
from .aerial import AerialData, video_timing
from .lower import LowerData


def frame_at_time(timestamps, time_ms):
    """Nearest presentation timestamp; never extend a view beyond its coverage."""
    times = np.asarray(timestamps)
    margin = float(np.median(np.diff(times)))/2 if len(times) > 1 else .5
    if time_ms < times[0]-margin or time_ms > times[-1]+margin:
        return None
    right = int(np.searchsorted(times, time_ms))
    choices = [i for i in (right-1, right) if 0 <= i < len(times)]
    return min(choices, key=lambda i: (abs(float(times[i])-time_ms), i))


def points(xy, confidence, valid):
    mask = valid.astype(bool) & np.isfinite(xy).all(axis=-1)
    return {'xy': [p.tolist() if ok else None for p, ok in zip(xy, mask)],
            'confidence': [float(v) if np.isfinite(v) else None for v in confidence],
            'valid': mask.tolist()}


class Clip:
    def __init__(self, path, video_root, row, view, model=None):
        with h5py.File(path) as h:
            self.source = json.loads(h.attrs['source_json'])
            if model is None:
                model = json.loads(h.attrs['model_json'])
            if h.attrs.get('schema_version') != '1.1':
                raise ValueError('UL/UR viewer requires schema 1.1, with optional normalized coordinates')
        meta = row['views'][view]
        self.video = (video_root/meta['relpath']).resolve()
        if not self.video.is_relative_to(video_root):
            raise ValueError('Source path is outside the video tree')
        expected = {'pair_id':row['pair_id'], 'pid':row['pid'], 'view':view, 'relpath':meta['relpath']}
        expected.update({key:meta[key] for key in ('width','height') if key in meta})
        for key, value in expected.items():
            if self.source[key] != value:
                raise ValueError('Source identity mismatch: '+key)
        validate_hdf5(path, source_sha256=sha256(self.video), model=model)
        with h5py.File(path) as h:
            if h.attrs['pose_topology'] != 'coco_wholebody_133' or h.attrs['face_topology'] != 'mediapipe_478':
                raise ValueError('Viewer requires explicitly named COCO WholeBody 133 / MediaPipe 478 topologies')
            self.times = h['frames/timestamp_ms'][:]
            self.pose = {key:h['participant/pose/'+key][:] for key in ('xy_px','confidence','valid')}
            self.face = {key:h['participant/face/'+key][:] for key in ('xy_px','confidence','valid')}
            self.hands = {key:(value.asstr()[:] if h5py.check_string_dtype(value.dtype) else value[:])
                          for key,value in h['hands'].items()}
            self.objects = {key:(value.asstr()[:] if h5py.check_string_dtype(value.dtype) else value[:])
                            for key,value in h['objects'].items()}
        self.capture = cv2.VideoCapture(str(self.video))
        if not self.capture.isOpened():
            raise ValueError('Cannot decode source video')
        self.next_index = 0
        self.cached_index, self.cached_image = None, None

    def close(self):
        self.capture.release()

    def image(self, index):
        if index == self.cached_index:
            return self.cached_image
        if index != self.next_index:
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError('Video seek failed')
        okay, image = self.capture.read()
        if not okay or int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) != index+1:
            raise ValueError('Decoded frame index differs from requested frame')
        if image.shape[:2] != (self.source['height'],self.source['width']):
            raise ValueError('Decoded source dimensions changed')
        self.next_index = index+1
        self.cached_index, self.cached_image = index, image
        return image

    def frame(self, index, include_image=True):
        if not 0 <= index < len(self.times):
            raise IndexError('Frame outside clip')
        result = {'frame_index':index,'timestamp_ms':float(self.times[index]),
                  'width':self.source['width'],'height':self.source['height'],
                  'pose':points(self.pose['xy_px'][index],self.pose['confidence'][index],self.pose['valid'][index]),
                  'face':points(self.face['xy_px'][index],self.face['confidence'][index],self.face['valid'][index]),
                  'hands':[], 'objects':[]}
        actors = {v:k for k,v in ACTORS.items()}; sides = {v:k for k,v in HANDEDNESS.items()}
        for i in np.flatnonzero(self.hands['frame_index'] == index):
            h = self.hands
            result['hands'].append({**points(h['xy_px'][i],h['confidence'][i],h['valid'][i]),
                'actor':actors[int(h['actor'][i])], 'handedness':sides[int(h['handedness'][i])],
                'track_id':int(h['track_id'][i]), 'topology':str(h['topology'][i]), 'model_id':str(h['model_id'][i]),
                'bbox':h['bbox_xyxy_px'][i].tolist() if h['bbox_valid'][i] else None,
                'detection_confidence':float(h['detection_confidence'][i]) if np.isfinite(h['detection_confidence'][i]) else None})
        for i in np.flatnonzero(self.objects['frame_index'] == index):
            o = self.objects
            result['objects'].append({'bbox':o['bbox_xyxy_px'][i].tolist(),'class_name':str(o['class_name'][i]),
                                      'confidence':float(o['confidence'][i]),'model_id':str(o['model_id'][i])})
        if include_image:
            okay, jpg = cv2.imencode('.jpg', self.image(index), [cv2.IMWRITE_JPEG_QUALITY,85])
            if not okay:
                raise ValueError('Frame encoding failed')
            result['image'] = 'data:image/jpeg;base64,'+base64.b64encode(jpg).decode('ascii')
        return result


class AerialClip(Clip):
    """Legacy reader sharing only the verified decoder with the UL/UR reader."""
    def __init__(self, path, video):
        self.video = video
        times, width, height = video_timing(video)
        self.data = AerialData(path, times, width, height)
        self.times = self.data.times
        self.source = {'width': width, 'height': height}
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise ValueError('Cannot decode AV source video')
        self.next_index = 0
        self.cached_index, self.cached_image = None, None

    def frame(self, index, include_image=True):
        result = self.data.frame(index)
        if include_image:
            okay, jpg = cv2.imencode('.jpg', self.image(index), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not okay:
                raise ValueError('AV frame encoding failed')
            result['image'] = 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode('ascii')
        return result


class LowerClip(AerialClip):
    """Legacy LL/LR reader; row indices are matched to decoded video PTS."""
    def __init__(self, path, video):
        self.video = video
        times, width, height = video_timing(video)
        self.data = LowerData(path, times, width, height, video_sha256=sha256(video))
        self.times = self.data.times
        self.source = {'width': width, 'height': height}
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise ValueError('Cannot decode lower-view source video')
        self.next_index = 0
        self.cached_index, self.cached_image = None, None


class Collection:
    def __init__(self, root, inventory, model_key, quality_root=None, aerial_root=None):
        self.root = root.resolve(); self.model_key = model_key
        scope = json.loads((self.root/'run_scope.json').read_text())
        if scope['manifest_sha256'] != sha256(inventory):
            raise ValueError('Inventory does not match the collection')
        self.locations = Locations(self.root, inventory)
        self.video_root = self.locations.video_root
        self.aerial_root = aerial_root.resolve() if aerial_root else self.locations.landmarks_root
        self.aerial_failures = {}
        self.model = json.loads((self.root/(model_key+'_configuration.json')).read_text())
        self.rows = json.loads(inventory.read_text())
        seen = set()
        for row in self.rows:
            path = Path(row['pair_id'])
            if path.is_absolute() or '..' in path.parts or row['pair_id'] in seen or set(row['views']) != {'CAM_UL','CAM_UR'}:
                raise ValueError('Invalid pair identity in inventory')
            seen.add(row['pair_id'])
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.quality_root = quality_root

    def clip(self, pair_index, view):
        if not 0 <= pair_index < len(self.rows) or view not in ('CAM_AV','CAM_UL','CAM_UR'):
            raise ValueError('Unknown pair or view')
        key = pair_index,view
        if key not in self.cache:
            row = self.rows[pair_index]
            if view == 'CAM_AV':
                if self.aerial_root is None:
                    raise ValueError('Aerial landmark directory is not configured')
                relative = Path(row['views']['CAM_UL']['relpath'].replace('-CAM_UL-', '-CAM_AV-'))
                self.cache[key] = AerialClip(contained(self.aerial_root, relative.with_suffix('.h5')),
                                             contained(self.video_root, relative))
            else:
                path = self.locations.output(row, view, self.model_key)
                self.cache[key] = Clip(path,self.video_root,row,view,self.model)
            while len(self.cache) > 6:
                _,old = self.cache.popitem(last=False); old.close()
        self.cache.move_to_end(key)
        return self.cache[key]

    def aerial(self, pair_index):
        if pair_index not in self.aerial_failures:
            try:
                return self.clip(pair_index, 'CAM_AV')
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                self.aerial_failures[pair_index] = 'Aerial view unavailable: ' + str(exc)
        return None

    def index(self):
        return {'pairs':[{'id':i,'pair_id':r['pair_id'],'participant':r['pid'],
                          'task':r.get('subtask_dir',Path(r['pair_id']).parent.as_posix())} for i,r in enumerate(self.rows)],
                'model':'RTMW-l · MediaPipe face · Hand5 · LEGO detector',
                'hand_refinement':bool(self.model.get('hand_refinement'))}

    def info(self, pair_index):
        with self.lock:
            views = {view:self.clip(pair_index,view) for view in ('CAM_UL','CAM_UR')}
            aerial = self.aerial(pair_index)
            if aerial is not None:
                views['CAM_AV'] = aerial
            return {'pair_id':self.rows[pair_index]['pair_id'],
                    'primary_view': 'CAM_UL',
                    'aerial_status': self.aerial_failures.get(pair_index,
                        'AV landmark timestamps checked against video PTS; playback follows relative clip time.'),
                    'views':{view:{'frames':len(clip.times),'timestamps_ms':clip.times.tolist(),
                                   'width':clip.source['width'],'height':clip.source['height']} for view,clip in views.items()}}

    def frame(self, pair_index, index, include_image=True):
        with self.lock:
            left = self.clip(pair_index,'CAM_UL')
            if not 0 <= index < len(left.times):
                raise ValueError('Frame outside clip')
            right = self.clip(pair_index,'CAM_UR')
            matched = frame_at_time(right.times,float(left.times[index]))
            aerial = self.aerial(pair_index)
            av_index = frame_at_time(aerial.times,float(left.times[index])) if aerial else None
            return {'CAM_UL':left.frame(index,include_image),
                    'CAM_AV':None if av_index is None else aerial.frame(av_index,include_image),
                    'CAM_UR':None if matched is None else right.frame(matched,include_image)}


class DatasetCollection(Collection):
    """Portable viewer backed only by videos/ and landmarks/ in a dataset copy."""
    def __init__(self, video_root, landmarks_root, quality_root=None):
        from .viewer_dataset import discover_dataset
        self.video_root, self.landmarks_root = video_root.resolve(), landmarks_root.resolve()
        self.rows, self.discovery_summary = discover_dataset(self.video_root, self.landmarks_root)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.quality_root = quality_root
        self.model = {}  # Each UL/UR file retains its own saved model provenance.
        self.view_failures = {}

    def clip(self, pair_index, view):
        if not 0 <= pair_index < len(self.rows) or view not in self.rows[pair_index]['views']:
            raise ValueError('Unknown clip or view')
        key = pair_index, view
        if key not in self.cache:
            row = self.rows[pair_index]; meta = row['views'][view]
            path = contained(self.landmarks_root, meta['landmark_relpath'])
            video = contained(self.video_root, meta['relpath'])
            if view == 'CAM_AV':
                self.cache[key] = AerialClip(path, video)
            elif view in ('CAM_LL', 'CAM_LR'):
                self.cache[key] = LowerClip(path, video)
            else:
                self.cache[key] = Clip(path, self.video_root, row, view)
            while len(self.cache) > 10:
                _, old = self.cache.popitem(last=False); old.close()
        self.cache.move_to_end(key)
        return self.cache[key]

    def loaded_views(self, pair_index):
        if not 0 <= pair_index < len(self.rows):
            raise ValueError('Unknown clip')
        views, errors = {}, {}
        for view in ('CAM_UL', 'CAM_AV', 'CAM_UR', 'CAM_LL', 'CAM_LR'):
            if view not in self.rows[pair_index]['views']:
                errors[view] = 'No matching video and landmark file for this view.'
                continue
            key = pair_index, view
            if key not in self.view_failures:
                try:
                    views[view] = self.clip(pair_index, view)
                except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
                    self.view_failures[key] = str(exc)
            if key in self.view_failures:
                errors[view] = self.view_failures[key]
        if not views:
            raise ValueError('No usable views for this clip: ' + '; '.join(f'{v}: {e}' for v,e in errors.items()))
        return views, errors

    def index(self):
        result = super().index()
        result.update(model='Stored AV/UL/UR/LL/LR landmarks', model_label='Saved landmarks · 2D source pixels',
                      discovery=self.discovery_summary)
        for entry, row in zip(result['pairs'], self.rows):
            entry['views'] = sorted(row['views'])
        return result

    def info(self, pair_index):
        with self.lock:
            views, errors = self.loaded_views(pair_index)
            return {'pair_id':self.rows[pair_index]['pair_id'], 'primary_view':next(iter(views)),
                    'view_errors':errors, 'aerial_status':errors.get('CAM_AV',
                        'Aerial timestamps checked against video; views follow relative clip time.'),
                    'views':{view:{'frames':len(clip.times),'timestamps_ms':clip.times.tolist(),
                                  'width':clip.source['width'],'height':clip.source['height']}
                             for view,clip in views.items()}}

    def frame(self, pair_index, index, include_image=True):
        with self.lock:
            views, _ = self.loaded_views(pair_index)
            primary = next(iter(views.values()))
            if not 0 <= index < len(primary.times):
                raise ValueError('Frame outside clip')
            time_ms = float(primary.times[index])
            result = {view: None for view in ('CAM_AV', 'CAM_UL', 'CAM_UR', 'CAM_LL', 'CAM_LR')}
            for view, clip in views.items():
                matched = frame_at_time(clip.times, time_ms)
                if matched is not None:
                    result[view] = clip.frame(matched, include_image)
            return result


def handler_for(collection):
    assets = { '/':'viewer.html', '/viewer.js':'viewer.js' }
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            host = self.headers.get('Host','')
            allowed = {'127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port)}
            origin = self.headers.get('Origin')
            if host not in allowed or (origin and origin != 'http://'+host) or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send_error(403); return
            parsed = urlparse(self.path)
            try:
                if parsed.path in assets:
                    data = Path(__file__).with_name(assets[parsed.path]).read_bytes()
                    kind = 'text/html; charset=utf-8' if parsed.path == '/' else 'text/javascript; charset=utf-8'
                else:
                    args = parse_qs(parsed.query)
                    if parsed.path == '/api/index':
                        result = collection.index()
                    elif parsed.path == '/api/clip':
                        result = collection.info(int(args['id'][0]))
                    elif parsed.path == '/api/frame':
                        result = collection.frame(int(args['id'][0]),int(args['frame'][0]),args.get('image',['1'])[0] != '0')
                    elif parsed.path == '/api/quality' and collection.quality_root and (collection.quality_root/'summary.json').exists():
                        result = json.loads((collection.quality_root/'summary.json').read_text())
                    else:
                        self.send_error(404); return
                    data = json.dumps(result,allow_nan=False).encode(); kind = 'application/json'
                self.send_response(200)
                self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers(); self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):
                return
            except (ValueError,KeyError,IndexError,OSError) as exc:
                self.send_error(400,str(exc))
        def log_message(self,*args):
            pass
    return Handler


def serve(collection, port=8767):
    try:
        server = ThreadingHTTPServer(('127.0.0.1',port),handler_for(collection))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE or getattr(exc, 'winerror', None) == 10048:
            alternative = 8770 if port != 8770 else 8771
            raise ValueError(
                f'Port {port} is already in use. If your viewer is already running, '
                f'open http://127.0.0.1:{port}/. To start another instance, run '
                f'python visualise.py --port {alternative}, or stop the previous server first.'
            ) from exc
        raise
    print(f'Local landmark viewer: http://127.0.0.1:{server.server_port}/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for clip in collection.cache.values():clip.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-root',required=True,type=Path)
    p.add_argument('--inventory',required=True,type=Path)
    p.add_argument('--quality-root',type=Path)
    p.add_argument('--aerial-landmarks-root', type=Path,
                   help='Defaults to the shared landmark root recorded in data_locations.json')
    p.add_argument('--model-key',default='rtmw_collection_v1_1')
    p.add_argument('--port',type=int,default=8767)
    a = p.parse_args()
    collection = Collection(a.collection_root,a.inventory,a.model_key,a.quality_root,a.aerial_landmarks_root)
    serve(collection, a.port)


if __name__ == '__main__':main()
