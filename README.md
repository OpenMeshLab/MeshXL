<!--     <img src="./assets/images/meshxl_logo.jpg" width="170px"> -->
<div align= "center">
    <h1> Official repo for MeshXL</h1>

</div>

<div align="center">
    <h2> <a href="https://arxiv.org/abs/2405.20853">MeshXL: Neural Coordinate Field for Generative 3D Foundation Models</a></h2>

<p align="center">
  <a href="https://meshxl.github.io/">Project Page</a> •
  <a href="https://arxiv.org/abs/2405.20853">Arxiv Paper</a> •
  <a href="">HuggingFace Demo</a> •
  <a href="#-citation">Citation
</p>

</div>

<div align="center">

<!-- <img src="https://cdn.discordapp.com/attachments/941582479117127680/1111543600879259749/20230526075532.png" width="350px"> -->

<!-- |                                                   Teaser Video                                                   |                                                    Demo Video                                                    |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/OpenMotionLab/MotionGPT/assets/120085716/a741e162-b2f4-4f65-af8e-aa19c4115a9e" /> | <video src="https://github.com/OpenMotionLab/MotionGPT/assets/120085716/ae966d17-6326-43e6-8d5b-8562cf3ffd52" /> | -->

</div>

<!-- ### [MeshXL: Neural Coordinate Field for Generative 3D Foundation Models](https://motion-gpt.github.io/) -->
<!-- ### [Project Page](https://motion-gpt.github.io/) | [Arxiv Paper](https://arxiv.org/abs/2306.14795) | [HuggingFace Demo](xxx) -->


## 🏃 Intro MeshXL

**MeshXL** is a family of generative pre-trained foundation models for 3D mesh generation. With the Neural Coordinate Field representation, the generation of unstructured 3D mesh data can be seaminglessly addressed by modern LLM methods.

The polygon mesh representation of 3D data exhibits great flexibility, fast rendering speed, and storage efficiency, which is widely preferred in various applications. However, given its unstructured graph representation, the direct generation of high-fidelity 3D meshes is challenging. Fortunately, with a pre-defined ordering strategy, 3D meshes can be represented as sequences, and the generation process can be seamlessly treated as an auto-regressive problem. In this paper, we validate the **Neur**al **C**oordinate **F**ield (NeurCF), an explicit coordinate representation with implicit neural embeddings, is a simple-yet-effective representation for large-scale sequential mesh modeling. After that, we present **MeshXL**, a family of generative pre-trained auto-regressive models, which addresses the process of 3D mesh generation with modern large language model approaches. Extensive experiments show that MeshXL is able to generate high-quality 3D meshes, and can also serve as foundation models for various down-stream applications.

<img width="1194" alt="pipeline" src="./assets/images/pipeline.png">


## 🚩 News

- [2024/07/23] Upload the inference code.
- [2024/06/02] Upload paper and init project.


## ⚡ Quick Start


<details>
  <summary><b>Environment Setting Up</b></summary>

  You can build the environment using the provided script:
  ```{bashrc}
  bash set_env.sh
  ```

</details>


<details>
  <summary><b>Data</b></summary>

  Work in progress...

</details>




## 💻 Training and Evaluation

<details>
  <summary><b>Download Pre-Trained Weights</b></summary>

  We provide pre-trained weights for different sizes of models (i.e. `125m`, `350m`, and `1.3b`) on huggingface. Download the pre-trained weights from the links below to replace the `pytorch_model.bin` files in the corresponding folders under the `./mesh-xl/` folder. The model details are shown below:

  <img width="500" alt="scaling" src="./assets/images/scaling.png">

  | Model Size |                    Download Link                    |
  |:----------:|:---------------------------------------------------:|
  |    125M    | [link](https://huggingface.co/CH3COOK/mesh-xl-125m) |
  |    350M    | [link](https://huggingface.co/CH3COOK/mesh-xl-350m) |
  |    1.3B    | [link](https://huggingface.co/CH3COOK/mesh-xl-1.3b) |



</details>


<details>
  <summary><b>MeshXL Generative Pre-Training</b></summary>

  Work in progress...

</details>


<details>
  <summary><b>Generating Samples</b></summary>

  <img width="1194" alt="samples" src="./assets/images/objaverse-samples.png">

  To generate 3D meshes with different sizes:
  ```{bashrc}
  bash scripts/sample-1.3b.sh
  bash scripts/sample-350m.sh
  bash scripts/sample-125m.sh
  ```

</details>



## 📖 Citation

If you find our code or paper helps, please consider citing:

```bibtex
@misc{chen2024meshxl,
      title={MeshXL: Neural Coordinate Field for Generative 3D Foundation Models}, 
      author={Sijin Chen and Xin Chen and Anqi Pang and Xianfang Zeng and Wei Cheng and Yijun Fu and Fukun Yin and Yanru Wang and Zhibin Wang and Chi Zhang and Jingyi Yu and Gang Yu and Bin Fu and Tao Chen},
      year={2024},
      eprint={2405.20853},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

Thanks to these amazing works: ShapeNet, 3D-FUTURE, Objaverse, and Objaverse-XL.
