import { add, map, multiply, transpose } from "./math";
import { Layer } from "./Layer";
import { Weights } from "./Weights";
import { sigmoidPrime } from "./utils";

export class Network {
  // X layers
  layers: Layer[] = [];

  // X-1 weights
  weights: Weights[] = [];

  // Learning rate
  lr: number = 0.1;

  constructor(numberOfLayers: number[], lr: number = 0.1) {
    let prevNumberOfNeurons;
    for (let numberOfNeurons of numberOfLayers) {
      this.layers.push(new Layer(numberOfNeurons));
      if (prevNumberOfNeurons) {
        this.weights.push(new Weights(numberOfNeurons, prevNumberOfNeurons));
      }
      prevNumberOfNeurons = numberOfNeurons;
    }
    this.lr = lr;
  }

  getLastLayer() {
    return this.layers[this.layers.length - 1];
  }

  getActivations() {
    return this.getLastLayer().getActivations();
  }

  guess(input: number[]) {
    this.layers[0].setActivations(input);
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].calculateActivations(
        this.layers[i - 1].activations,
        this.weights[i - 1].weights
      );
    }
  }

  getCost(desiredOutput: number[]) {
    let C = 0;
    const activations = this.getActivations();

    // Cost: C = sum( (aj - yj)² )
    for (let j = 0; j < activations.length; j++) {
      const diff = activations[j] - desiredOutput[j];
      C += Math.pow(diff, 2);
    }

    return C;
  }

  train(inputs: number[][], desiredOutputs: number[][]): number {
    let dWsSum: number[][] = [];
    let dBsSum: number[] = [];
    let costSum: number = 0;
    const length = inputs.length;

    for (let i = 0; i < length; i++) {
      this.guess(inputs[i]);
      const { dWs, dBs } = this.getDeltas(desiredOutputs[i]);
      for (let k = 0; k < dWs.length; k++) {
        dWsSum[k] = dWs[k].map((dW, j) => dW + (dWsSum[k] ? dWsSum[k][j] || 0 : 0));
      }
      dBsSum = dBs.map((dB, j) => dB + (dBsSum[j] || 0));
      costSum += this.getCost(desiredOutputs[i]);
    }

    // Calculate the mediums
    const dWsMoy = [];
    for (let k = 0; k < dWsSum.length; k++) {
      dWsMoy[k] = dWsSum[k].map((dWSum) => dWSum / length);
    }
    const dBsMoy = dBsSum.map((dBSum) => dBSum / length);
    const costMoy = costSum / length;

    // Apply new weight
    const weights = this.weights[0].weights;
    for (let k = 0; k < weights.length; k++) {
      for (let j = 0; j < weights[0].length; j++) {
        weights[k][j] += dWsMoy[k][j] * this.lr;
      }
    }

    // Apply new biases
    const biases = this.layers[1].biases;
    for (let i = 0; i < biases.length; i++) {
      biases[i][0] += dBsMoy[i] * this.lr;
    }

    return costMoy;
  }

  getDeltas(desiredOutput: number[]): { dWs: number[][]; dBs: number[] } {
    const l = this.layers.length - 1;

    // Desired output (Column matrice)
    const y: [number][] = desiredOutput.map((result) => [result]);

    // Impact on cost relative to activation (Column matrice)
    const dCdAl: [number][] = multiply(
      2,
      add(
        y,
        map(this.layers[l].activations, (x) => -x)
      )
    ) as [number][];

    // Impact on activation relative to Z (Column matrice)
    const dAldZl: [number][] = map(
      this.layers[l].getZ(
        this.layers[l - 1].activations,
        this.weights[l - 1].weights
      ),
      sigmoidPrime
    );

    // Impact on Z relative to weights (Column matrice)
    const dZldWl: [number][] = this.layers[l - 1].activations;

    // Impact on Z relative to biases
    const dZldBl = 1;

    // Impact on Z relative to previous activation
    // const dZldAlminus1: number[][] = this.weights[l - 1].weights;

    // Common term for next calculations: delta = dAldZl . dCdAl
    // const delta = multiply(transpose(dAldZl), dCdAl);
    // console.log(delta);

    // Impact on cost relative to weights: dCdWl = dZldWl . dAldZl . dCdAl = dZldWl . delta
    // const dCdWl = multiply(multiply(dZldWl, transpose(dAldZl)), dCdAl);
    // console.log(dCdWl);

    const dWs = [];
    const dBs = [];
    const weights = this.weights[l - 1].weights;
    for (let j = 0; j < weights[0].length; j++) {
      const delta = dAldZl[j][0] * dCdAl[j][0];
      dBs.push(dZldBl * delta);
      for (let k = 0; k < weights.length; k++) {
        dWs[k] ||= [];
        dWs[k][j] = dZldWl[k][0] * delta;
      }
    }

    return { dWs, dBs };
  }
}
