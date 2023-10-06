Dataset **Kvasir Instrument** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/L/e/IW/zBZlGPC4PQ2tySAOCyf0fWwvRSjGp3YVUvHlwtB5zCbwW2BCthMsrd1qABrFyC0fUgdc3yevhr49ZwjUxkdjBC1UFwPnChX49kW33Ucd2FdSa0DBN0gVBo0HfpQz.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Kvasir Instrument', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://datasets.simula.no/downloads/kvasir-instrument.zip).