import { draw } from './draw';
import { Network } from "./Network";

import "./style.css";

const network = new Network([3, 6]);
const output = network.getOutput([1, 0, 0]);
const max = Math.max(...output);
const classificationIndex = output.indexOf(max);

const inputLabels = ['Red', 'Green', 'Blue'];
const outputLabels = ['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta'];
draw(network, inputLabels, outputLabels);

document.getElementById('app').innerHTML = `Result is : ${outputLabels[classificationIndex]}`
