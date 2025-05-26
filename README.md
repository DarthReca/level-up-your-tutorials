<div align="center">
  
## Level Up Your Tutorials: VLMs for Game Tutorials Quality Assessment

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Gabriele Scaffidi**](https://smartdata.polito.it/members/gabriele-scaffidi-militone/)<sup>1</sup> · [**Luca Colomba**](https://github.com/lccol)<sup>1</sup>

[**Giovanni Malnati**](https://www.polito.it/en/staff?p=giovanni.malnati)<sup>1</sup> · [**Daniele Apiletti**](https://www.polito.it/en/staff?p=daniele.apiletti)<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy

**[ECCV 2024 CV2 Workshop](https://sites.google.com/nvidia.com/cv2/)**

<a href="https://arxiv.org/abs/2408.08396"><img src='https://img.shields.io/badge/ArXiv-Level%20Up%20Your%20Tutorials-red?logo=arxiv' alt='Paper PDF'></a>
<a href='https://huggingface.co/datasets/DarthReca/but-they-are-cats-tutorial'><img src='https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?logo=huggingface'></a>
</div>

**In this work, we propose an automated game-testing solution to evaluate the quality of game tutorials.** Our approach leverages VLMs to analyze frames from video game tutorials, answer relevant questions to simulate human perception, and provide feedback. This feedback is compared with expected results to identify confusing or problematic scenes and highlight potential errors for developers.

### Getting Started

Install the dependencies for the model as declared in the respective repositories. Use *generate.ipynb* to generate the answers using a desired model. 
The *models folder* contains wrappers for the tested models.

## Dataset

The dataset is available on [HuggingFace](https://huggingface.co/datasets/DarthReca/but-they-are-cats-tutorial). For more information about the data, refer to the HuggingFace page.

## Contributors
The repository setup is by [Luca Colomba](https://github.com/lccol) and [Daniele Rege Cambrin](https://github.com/DarthReca).

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inbook{RegeCambrin2025,
  title = {Level Up Your Tutorials: VLMs for Game Tutorials Quality Assessment},
  ISBN = {9783031923876},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-92387-6_26},
  DOI = {10.1007/978-3-031-92387-6_26},
  booktitle = {Computer Vision – ECCV 2024 Workshops},
  publisher = {Springer Nature Switzerland},
  author = {Rege Cambrin,  Daniele and Scaffidi Militone,  Gabriele and Colomba,  Luca and Malnati,  Giovanni and Apiletti,  Daniele and Garza,  Paolo},
  year = {2025},
  pages = {374–389}
}
```
