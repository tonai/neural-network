import { draw } from './draw';
import { Network } from "./Network";

import "./style.css";
import { format } from './utils';

const inputLabels = ['Red', 'Green', 'Blue'];
const inputs = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
];

const outputLabels = ['Red', 'Green', 'Blue'];
const desiredOutputs = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
];

const network = new Network([3, 3], 5);
draw(network, inputLabels, outputLabels);

const costEl = document.getElementById('cost');
const trainel = document.getElementById('train');

trainel.addEventListener('click', () => {
  const averageCost = network.train(inputs, desiredOutputs);
  costEl.innerText = format(averageCost).toString();
});
