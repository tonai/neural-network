import { rand } from './utils';

export class Weights {
  // Two dimension matrice: [[w0,0 w0,1 ... w0,n] [w1,0 w1,1 ... w1,n] ... [wk,0 wk,1 ... wk,n]]
  weights: number[][] = [];

  constructor(k: number, n: number) {
    for (let i = 0; i < k; i++) {
      this.weights.push([]);
      for (let j = 0; j < n; j++) {
        this.weights[i].push(rand());
      }
    }
  }
}