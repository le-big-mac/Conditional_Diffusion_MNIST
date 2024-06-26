# Variational continual learning for diffusion models.

A codespace for training diffusion models with [variational continual learning](https://arxiv.org/abs/1710.10628), to see if it can mitigate catastrophic forgetting.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="images/standard.png" width="200" />
        <br>
        <em>Standard training.</em>
      </td>
      <td align="center">
        <img src="images/vcl.png" width="200" />
        <br>
        <em>VCL training.</em>
      </td>
    </tr>
    <tr>
      <td colspan="2" align="center">
        <em>Images generated during continual learning of MNIST digits, with no data replay. Each row i shows generations from when the model had been trained on
            the first i tasks (digits), each column j shows generations for digit j.</em>
      </td>
    </tr>
  </table>
</div>

Note that the variational inference for VCL is very expensive, so this code will take several hours to run on a powerful GPU.

This diffusion model in this project is based on original work by Tim Pearce, released under the MIT License in 2022. The original code can be found at [Conditional Diffusion MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST). The modifications made by me in 2024 are also released under the MIT License.
