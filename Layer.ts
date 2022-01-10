import { add, map, multiply } from 'mathjs';

// import { Neuron } from './Neuron';
import { rand, sigmoid } from './utils';

export class Layer {
  // Column matrice: [[a0] [a1] ... [an]]
  // neurons: [Neuron][] = [];
  activations: [number][] = [];

  // Column matrice: [[b0] [b1] ... [bn]]
  biases: [number][] = [];

  constructor(numberOfNeurons: number) {
    for(let i = 0; i < numberOfNeurons; i++) {
      // this.neurons.push([new Neuron()]);
      this.activations.push([0]);
      this.biases.push([rand()]);
    }
  }

  setActivations(input: number[]) {
    for (let i = 0; i < input.length; i++) {
      // this.neurons[i][0].activation = input[i];
      this.activations[i][0] = input[i];
    }
  }

  getActivations(): number[] {
    // return this.neurons.map(([neuron]) => neuron.activation);
    return this.activations.map(([activation]) => activation);
  }

  calculateActivations(activations: [number][], weights: number[][]) {
    this.activations = map(add(multiply(weights, activations), this.biases) as [number][], sigmoid);
  }
}
