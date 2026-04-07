# Large language model-based evaluation of the impact of gender in medical research

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

Gender disparities in academic medicine have been previously reported, but prior bibliometric studies have been limited by small sample sizes and reliance on manual gender annotation methods. These bottlenecks constrain previous analyses to only a small subset of clinical literature. To assess gender-based differences in authorship trends, research impact, and scholarly output over time in clinical research at scale, we hypothesized that large language models (LLMs) can be an effective tool to facilitate systematic bibliometric analysis of academic research trends. We conducted a retrospective, cross-sectional bibliometric study evaluating manuscripts published between January 2015 and September 2025 across over 1,000 PubMed-indexed academic medical journals. Over 1 million manuscripts, written by more than 10 million authors across 13 medical specialties, were analyzed. To enable this large-scale study, the genders of manuscript authors were annotated using a scalable LLM-based pipeline compatible with consumer-grade hardware.

As a part of this project, we have created the `namecast` package, which provides a standardized API to make gender predictions from both LLMs and conventional database-based methods.

## Installation and Usage

If you are interested in using the `namecast` package, all you need to do is install it using `pip`:

```
python -m pip install namecast
```

To see what gender prediction methods are natively available with `namecast`, you can run

```python
import namecast
print(namecast.list_registered_methods())
```

You can choose any of the listed gender prediction methods to instantiate a gender prediction engine, which can then be used for generating gender predictions:

```python
engine = namecast.make("meta-llama/Llama-3.1-8B")
assert "female" == engine.predict("Alice")  # Predicts the gender of a single name.
assert ["female", "male"] == engine.predict_batch(["Alice", "Bob"])  # Predicts the gender of a batch of names.
```

If you are interested in reproducing our research based on the `namecast` package, we have provided a Dockerfile that specifies the expected compute environment. You can first build an image and then run a corresponding container using:

```
docker build -t medicine-authorship:latest .
docker run -it medicine-authorship:latest bash
```

To reproduce our experimental results, all you need to do is run

```
bash run.sh
```

Please refer to the [`scripts`](./scripts) directory for the individual script file implementations.

## Contact

Questions and comments are welcome. Suggestions can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

## Citation

If you found our work helpful for your research, please consider citing our paper:

    @misc{yaoms2026medanalysis,
      title={Large language model-based evaluation of the impact of gender in medical research},
      author={Yao, Michael S},
      year={2026},
      doi={10.64898/2026.01.06.26343564},
      url={https://www.medrxiv.org/node/1135425.full}
    }

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
