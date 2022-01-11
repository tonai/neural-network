import { rand } from './utils';

export class Weights {
  // Two dimension matrice: [[w0,0 w0,1 ... w0,n] [w1,0 w1,1 ... w1,n] ... [wk,0 wk,1 ... wk,n]]
  // Lines represents k (layer), columns represents n (previous layer)
  weights: number[][] = [];

  constructor(prevN: number, n: number) {
    for (let k = 0; k < prevN; k++) {
      this.weights.push([]);
      for (let j = 0; j < n; j++) {
        this.weights[k].push(rand());
      }
    }
  }
}