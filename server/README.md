### Creating a Virtual Environment

To create a virtual environment in Python, use the following command:

```bash
python -m venv myenv
```

Replace `myenv` with the desired name for your virtual environment. To activate the virtual environment, use the following command:

On Windows:
```bash
myenv\Scripts\activate
```

On macOS and Linux:
```bash
source myenv/bin/activate
```

To deactivate the virtual environment, simply run:
```bash
deactivate
```


### Docker commands

Enable Docker Buildx to create a multi-architecture image:

```bash
docker buildx create --use
docker buildx inspect --bootstrap
```

To build for a specific architecture (amd64) and push the image to Docker Hub, use the following command:

```bash
docker buildx build --platform linux/amd64 -t afscomercial/ctquarterreportscrapping:latest --push .
```