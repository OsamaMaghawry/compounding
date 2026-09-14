# B-roll library

Drop your own clips and stills here. Keywords come from the filename, so Arabic
filenames work directly:

    فلوس.mp4          -> matches فلوس / الفلوس
    الذكاء_الاصطناعي.mp4 -> matches الذكاء or الاصطناعي
    laptop-work.mp4    -> matches laptop or work

For several keywords per clip, or to skip into an asset, add `library.yml` here:

```yaml
money.mp4:
  keywords: [فلوس, مال, ارباح, ربح]
  start: 1.5      # start 1.5s into the clip
laptop.mp4: [لابتوب, كمبيوتر, شغل]
```

Matching is done on normalised Arabic, so `الذّكاء` and `الذكاء` both hit.
Assets you keep in review get promoted over time; ones you delete sink.
