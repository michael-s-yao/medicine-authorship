# Large language model-based evaluation of the impact of gender in medical research

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

The purpose of this research is to assess gender-based differences in authorship trends, research impact, and scholarly output in medical research using LLMs. This retrospective study evaluated manuscripts published between January 2015 and September 2025 across 184 PubMed-indexed radiology journals. Over 1 million manuscripts comprising over 10 million authors were analyzed. Author gender was annotated using a scalable LLM-based pipeline compatible with consumer grade hardware. We queried an LLM to generate multiple independent gender predictions per author, with majority agreement used to define the final classification. We found that the proportion of female authors generally increased over time. Our results demonstrate a scalable, automated method to track bibliometric trends in academic research, highlighting both progress and persistent gender inequaities in the authorship of medical literature.

## Installation

To install and run our code, first clone the `radiology-authorship` repository.

```
$ cd ~
$ git clone https://github.com/michael-s-yao/medicine-authorship
$ cd medicine-authorship
```

To maximize reproducibility of our work, we have provided a Dockerfile that specifies the expected compute environment. You can first build an image and then run a corresponding container using:

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

    @misc{yaoms2025radanalysis,
      title={Large language model-based evaluation of the impact of gender in radiological research},
      author={Yao, Michael S},
      year={2025}
    }

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
