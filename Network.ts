import { Layer } from "./Layer";
import { Weights } from "./Weights";

export class Network {
  // X layers
  layers: Layer[] = [];

  // X-1 weights
  weights: Weights[] = [];

  constructor(numberOfLayers: number[]) {
    let prevNumberOfNeurons;
    for(let numberOfNeurons of numberOfLayers) {
      this.layers.push(new Layer(numberOfNeurons));
      if (prevNumberOfNeurons) {
        this.weights.push(new Weights(numberOfNeurons, prevNumberOfNeurons));
      }
      prevNumberOfNeurons = numberOfNeurons;
    }
  }

  getOutput(input: number[]) {
    this.layers[0].setActivations(input);
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].calculateActivations(this.layers[i - 1].activations, this.weights[i - 1].weights);
    }
    return this.layers[this.layers.length - 1].getActivations();
  }
}