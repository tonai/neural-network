import { draw } from './draw';
import { Network } from "./Network";

import "./style.css";
import { convertToDecimal, format } from './utils';

const inputLabels = ['Red', 'Green', 'Blue'];
const inputs = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 1],
  [1, 0, 1]
];

const outputLabels = ['Red', 'Green', 'Blue', 'yellow', 'cyan', 'magenta'];
const desiredOutputs = [
  [1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1]
];

const network = new Network([3, 4, 6], 5);
draw(network, inputLabels, outputLabels);

const colorEl = document.getElementById('color') as HTMLInputElement;
const costEl = document.getElementById('cost');
const guessEl = document.getElementById('guess');
const resultEl = document.getElementById('result');
const trainEl = document.getElementById('train');

trainEl.addEventListener('click', () => {
  const averageCost = network.train(inputs, desiredOutputs);
  costEl.innerText = format(averageCost).toString();
});

guessEl.addEventListener('click', () => {
  const color = convertToDecimal(colorEl.value);
  network.guess(color);
  const results = network.getActivations();
  const max = Math.max(...results);
  const resultIndex = results.indexOf(max);
  resultEl.innerText = outputLabels[resultIndex];
});
