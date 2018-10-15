# MANUAL REPRODUCTION
From directory containing CI scripts execute runCI.sh bash script:

```
cd <path-to-onnx-repo>/.ci/jenkins/
./runCI.sh
```

To remove all items created during script execution (files, directories, docker images and containers), run:

```
./runCI.sh --cleanup
```

You can specify nGraph commit to run tests on, e.g.:

```
./runCI.sh --ngraph-commit=r8nd0mn6r4phc00mm1t5h4
```
