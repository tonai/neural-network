import { add, map, multiply } from 'mathjs';

import { rand, sigmoid } from './utils';

export class Layer {
  // Column matrice: [[a0] [a1] ... [an]]
  activations: [number][] = [];

  // Column matrice: [[b0] [b1] ... [bn]]
  biases: [number][] = [];

  constructor(numberOfNeurons: number) {
    for(let j = 0; j < numberOfNeurons; j++) {
      this.activations.push([0]);
      this.biases.push([rand()]);
    }
  }

  setActivations(input: number[]) {
    for (let j = 0; j < input.length; j++) {
      this.activations[j][0] = input[j];
    }
  }

  getActivations(): number[] {
    return this.activations.map(([activation]) => activation);
  }

  getZ(activations: [number][], weights: number[][]): [number][] {
    return add(multiply(weights, activations), this.biases) as [number][];
  }

  calculateActivations(activations: [number][], weights: number[][]) {
    this.activations = map(this.getZ(activations, weights), sigmoid);
  }
}
