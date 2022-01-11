import P5 from "p5";

import { Network } from "./Network";
import { format } from "./utils";

interface Config {
  width: number
  height: number
  padding: number
  radius: number
}

const defaultConfig: Config = {
  width: 600,
  height: 600,
  padding: 20,
  radius: 20
}

export function draw(network: Network, inputLabels?: string[], outputLabels?: string[], config?: Partial<Config>) {
  const { width, height, padding, radius } = { ...defaultConfig, ...config };

  function sketch(p5: P5) {
    p5.setup = () => p5.createCanvas(width, height);

    p5.draw = () => {
      const numberOfLayers = network.layers.length;
      const maxNeuronInLayer = network.layers.reduce(
        (acc, layer) => Math.max(acc, layer.activations.length),
        -Infinity
      );
      const maxWeight = network.weights.reduce(
        (acc, weight) =>
          Math.max(
            acc,
            weight.weights.reduce(
              (acc, row) =>
                Math.max(
                  acc,
                  row.reduce(
                    (acc, weight) => Math.max(acc, Math.abs(weight)),
                    -Infinity
                  )
                ),
              -Infinity
            )
          ),
        -Infinity
      );

      const innerWidth = width - 2 * padding - 2 * radius;
      const innerHeight = height - 2 * padding - 2 * radius;
      const widthStep = numberOfLayers > 1 ? innerWidth / (numberOfLayers - 1) : 0;
      const heightStep = maxNeuronInLayer > 1 ? innerHeight / (maxNeuronInLayer - 1) : 0;

      p5.clear();
      p5.background(51);
      p5.textAlign(p5.CENTER, p5.BOTTOM);

      network.weights.forEach((weight, layerIndex) => {
        weight.weights.forEach((row, neuronInNextLayer, weights) => {
          const x0 = padding + radius + widthStep * (layerIndex + 1);
          const y0 =
            padding +
            radius +
            heightStep * neuronInNextLayer +
            innerHeight / 2 -
            (heightStep * (weights.length - 1)) / 2;

          row.forEach((weight, neuronInPrevLayer, row) => {
            const x1 = padding + radius + widthStep * layerIndex;
            const y1 =
              padding +
              radius +
              heightStep * neuronInPrevLayer +
              innerHeight / 2 -
              (heightStep * (row.length - 1)) / 2;

            p5.noFill();
            if (weight < 0) {
              p5.stroke(255, 0, 0);
            } else {
              p5.stroke(0, 255, 0);
            }
            p5.strokeWeight((Math.abs(weight) / maxWeight) * 10);
            p5.line(x0, y0, x1, y1);
          });
        });
      });

      network.layers.forEach((layer, layerIndex) => {
        layer.activations.forEach(([activation], neuronIndex, activations) => {
          const x = padding + radius + widthStep * layerIndex;
          const y =
            padding +
            radius +
            heightStep * neuronIndex +
            innerHeight / 2 -
            (heightStep * (activations.length - 1)) / 2;

          p5.fill(activation * 255);
          p5.stroke(255);
          p5.strokeWeight(1);
          p5.ellipse(x, y, 2 * radius, 2 * radius);

          p5.fill(activation > 0.5 ? 0 : 255);
          p5.noStroke();
          p5.text(format(activation), x, y + 7);
        });
      });

      function drawLabels(labels: string[], layerIndex: number) {
        if (labels) {
          labels.forEach((label, index) => {
            const x = padding + radius + widthStep * layerIndex;
            const y =
              padding +
              radius +
              heightStep * index +
              innerHeight / 2 -
              (heightStep * (network.layers[layerIndex].activations.length - 1)) / 2;
    
            p5.fill(255);
            p5.noStroke();
            p5.text(label, x, y - radius);
          });
        }
      }

      drawLabels(inputLabels, 0);
      drawLabels(outputLabels, numberOfLayers - 1);
    };
  }

  new P5(sketch);
}
