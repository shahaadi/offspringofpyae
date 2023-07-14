import random

# create a dictionary of songs and their metadata with their keys being their ids
class MetaData:
  count = 0
  songList = {}

  def __init__(self, name, artist, samples, sample_rate=44100):
    self.name = name
    self.artist = artist
    self.samples = samples
    self.sample_rate = sample_rate
    MetaData.count += 1
    self.id = MetaData.count
    MetaData.songList[self.id] = self

  def __str__(self):
    # return id
    return self.id + ": " + self.name + " by " + self.artist

  # all getter methods

  def getName(self):
    return self.name

  def getArtist(self):
    return self.artist

  def getSamples(self):
    return self.samples

  def getSampleRate(self):
    return self.sample_rate

  def getID(self):
    return self.id

  # function to create a smaller array of samples of a desired length (in time) from a larger array of samples, for further testing
  def small_sample(self, duration):
    n_samples = self.sample_rate * duration
    init = random.randint(0, len(self.samples) - n_samples)
    return self.samples[init:init + n_samples]

  @staticmethod
  def getSong(song_id):
    return MetaData.songList[song_id]
