import numpy as np
import pickle
from metadata import MetaData
from collections import Counter

class Database:
    def __init__(self):
        self._db = None
    def get_song(self, song_id):
        assert isinstance(self._db, dict), 'Load a database first. Use load_db to load a database'

        if song_id in self._db:
            return self._db[song_id]  # Return the song data associated with the song ID
        else:
            return None  # Return None if the song ID is not found in the database

    def load_db(self, fpath: str) -> None:
        assert self._db == None, 'Database already loaded. Use switch_db to switch databases'
        assert isinstance(fpath, str), 'Fpath must be of string type'

        with open(fpath, mode="rb") as open_file:
            db = pickle.load(open_file)
            assert isinstance(db, dict), 'Load a pickled dictionary for database'
            self._db = db
        return None

    def save_db(self, db: dict, fpath: str) -> None:
        assert isinstance(self._db, dict), 'Load a database first. Use load_db to load a database'
        assert isinstance(fpath, str), 'Fpath must be of string type'

        with open(fpath, mode="wb") as open_file:
            pickle.dump(self._db, open_file)
        return None

    def switch_db(self, fpath: str) -> None:
        assert isinstance(self._db, dict), 'Load a database first. Use load_db to load a database'
        assert isinstance(fpath, str), 'Fpath must be of string type'
        self.load_db(fpath)
        return None

    def del_song(self, song: MetaData) -> None:
        assert isinstance(self._db, dict), 'Load a database first. Use load_db to load a database'
        assert isinstance(song, MetaData), 'Song must be of MetaData type'

        for key in self._db.keys():
            self._db[key] = self._db[key][self._db[key][:, 0] != song.id]
        return None

    def add_song(self, song: MetaData, fingerprint: np.ndarray) -> None:
        assert isinstance(fingerprint, np.ndarray), 'Fingerprint must be of np.ndarray type'
        assert isinstance(song, MetaData), 'Song must be of MetaData type'

        if self._db == None:
            self._db = {}

        assert isinstance(self._db, dict), 'Dictionary is the only acceptable Database type'

        PEAK_PAIR_LEN = 3
        FANOUT_T_IDX = 3

        for fanout in fingerprint:
            for peak_data in fanout:
                fanout_t = peak_data[FANOUT_T_IDX]
                peak_pair = tuple(peak_data[:PEAK_PAIR_LEN])
                if peak_pair in self._db:
                    self._db[peak_pair] = np.append(self._db[peak_pair], [[song.id, fanout_t]], axis=0)
                else:
                    self._db[peak_pair] = np.array([[song.id, fanout_t]])

    def query_song(self, fingerprint: np.ndarray) -> int:
        assert isinstance(self._db, dict), 'Load a database first. Use load_db to load a database'
        assert isinstance(fingerprint, np.ndarray), 'Fingerprint must be of np.ndarray type'

        PEAK_PAIR_LEN = 3
        FANOUT_T_IDX = 3

        c = Counter()

        for fanout in fingerprint:
            for peak_data in fanout:
                fanout_t = peak_data[FANOUT_T_IDX]
                peak_pair = tuple(peak_data[:PEAK_PAIR_LEN])
                if peak_pair in self._db:
                    offset_songs = self._db[peak_pair].copy()
                    offset_songs[:, 1] -= fanout_t
                    c.update(list(map(tuple, offset_songs)))
        
        return c.most_common(1)[0][0][0]

    def get_db(self) -> dict:
        assert isinstance(self._db, dict), 'Load a database first. Use load_db to load a database'
        return self._db
