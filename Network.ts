import { add, map, multiply } from "./math";
import { Layer } from "./Layer";
import { Weights } from "./Weights";
import { sigmoidPrime } from "./utils";

interface Model {
  weights: number[][][]
  biases: [number][][]
}

export class Network {
  // X layers
  layers: Layer[] = [];

  // X-1 weights
  weights: Weights[] = [];

  // Learning rate
  lr: number = 1;

  constructor(numberOfLayers: number[], lr: number = 1) {
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
    let dWsSum: number[][][] = [];
    let dBsSum: number[][] = [];
    let costSum: number = 0;
    const length = inputs.length;

    for (let n = 0; n < length; n++) {
      this.guess(inputs[n]);
      const { dWs, dBs } = this.getDeltas(desiredOutputs[n]);
      for (let i = 0; i < dWs.length; i++) {
        for (let j = 0; j < dWs[i].length; j++) {
          dWsSum[i] ||= [];
          dWsSum[i][j] = dWs[i][j].map((dW, k) => dW + (dWsSum[i][j] ? dWsSum[i][j][k] || 0 : 0));
        }
      }
      for (let i = 0; i < dBs.length; i++) {
        dBsSum[i] = dBs[i].map((dB, j) => dB + (dBsSum[i] ? dBsSum[i][j] || 0 : 0));
      }
      costSum += this.getCost(desiredOutputs[n]);
    }

    // Calculate the mediums
    const dWsMoy: number[][][] = [];
    const dBsMoy: number[][] = [];
    for (let i = 0; i < dWsSum.length; i++) {
      for (let j = 0; j < dWsSum[i].length; j++) {
        dWsMoy[i] ||= [];
        dWsMoy[i][j] = dWsSum[i][j].map((dWSum) => dWSum / length);
      }
    }
    for (let i = 0; i < dBsSum.length; i++) {
      dBsMoy[i] = dBsSum[i].map((dBSum) => dBSum / length);
    }
    const costMoy = costSum / length;

    // Apply new weight
    for (let i = 0; i < this.layers.length - 1; i++) {
      const weights = this.weights[i].weights;
      for (let j = 0; j < weights.length; j++) {
        for (let k = 0; k < weights[j].length; k++) {
          weights[j][k] += dWsMoy[i][j][k] * this.lr;
        }
      }
    }

    // Apply new biases
    for (let i = 0; i < this.layers.length - 1; i++) {
      const biases = this.layers[i + 1].biases;
      for (let j = 0; j < biases.length; j++) {
        biases[j][0] += dBsMoy[i][j] * this.lr;
      }
    }

    return costMoy;
  }

  getDeltas(desiredOutput: number[]): { dWs: number[][][]; dBs: number[][] } {
    let l = this.layers.length - 1;

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

    // Impact on cost relative to weights: dCdWl = dZldWl . dAldZl . dCdAl = dZldWl . delta
    const dWs = [];

    // Impact on cost relative to biases: dCdBl = dZldBl . dAldZl . dCdAl = dZldBl . delta
    const dBs = [];

    this.getLayerDelta(l, dCdAl.map(([x]) => x), dWs, dBs);

    return { dWs, dBs };
  }

  getLayerDelta(l: number, dCdAl: number[], dWs: number[][][], dBs: number[][]) {
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
    const dZldAlminus1: number[][] = this.weights[l - 1].weights;
 
    // Impact on cost relative to previous activation: dCdAl-1 = dZldAl-1 . dAldZl . dCdAl = dZldAl-1 . delta
    const dCdAlminus1: number[] = [];
 
    const weights = this.weights[l - 1].weights;
    for (let j = 0; j < weights.length; j++) {
      const delta = dAldZl[j][0] * dCdAl[j];
      // dWs
      for (let k = 0; k < weights[j].length; k++) {
        dWs[l - 1] ||= [];
        dWs[l - 1][j] ||= [];
        dWs[l - 1][j][k] = dZldWl[k][0] * delta;
      }
      // dBs
      dBs[l - 1] ||= [];
      dBs[l - 1].push(dZldBl * delta);
      // dAs
      for (let k = 0; k < weights[j].length; k++) {
        dCdAlminus1[k] = (dCdAlminus1[k] || 0) + dZldAlminus1[j][k] * delta;
      }
    }

    if (l - 1 > 0) {
      this.getLayerDelta(l - 1, dCdAlminus1, dWs, dBs);
    }
  }

  export(): Model {
    const weights: number[][][] = [];
    const biases: [number][][] = [];
    for (let i = 0; i < this.weights.length; i++) {
      weights[i] = this.weights[i].weights;
    }
    for (let i = 0; i < this.layers.length; i++) {
      biases[i] = this.layers[i].biases;
    }
    return { weights, biases };
  }

  import(model: Model) {
    for (let i = 0; i < model.weights.length; i++) {
      this.weights[i].weights = model.weights[i];
    }
    for (let i = 0; i < model.biases.length; i++) {
      this.layers[i].biases = model.biases[i];
    }
  }
}
