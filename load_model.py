import gdown

id = "1NhJ-MTNRVOxH6uHR3KiDC6cLdN8ZnqH2"
output = "model_best.pth"
gdown.download(id=id, output=output, quiet=False)