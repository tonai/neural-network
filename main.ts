import mnist from 'mnist';
import { shuffle } from './utils';
import { Network } from "./Network";

import "./style.css";
import { format } from './utils';

const { training, test } = mnist.set(10000, 256);

const outputLabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

const network = new Network([784, 16, 16, 10]);
// draw(network, inputLabels, outputLabels);

const trainEl = document.getElementById('train');
const costEl = document.getElementById('cost');
const numberEl = document.getElementById('number');
const canvasEl = document.getElementById('canvas') as HTMLCanvasElement;
const guessEl = document.getElementById('guess');
const resultEl = document.getElementById('result');
const exportEl = document.getElementById('export');
const fileEl = document.getElementById('file') as HTMLInputElement;
const importEl = document.getElementById('import');

numberEl.innerHTML = '<option></option>' + test.map((_, i) => `<option value="${i}">${i}</option>`).join('');

const loop = 100;
const subsetLength = 100;
trainEl.addEventListener('click', () => {
  for (let i = 0; i < loop; i++) {
    const array = shuffle(training);
    console.log(`training ${array.length} items`);
    let averageCost = 0;
    let j = 0;
    for (j = 0; j < array.length; j += subsetLength) {
      const trainingSubset = array.slice(j, j + subsetLength);
      const inputs = trainingSubset.map(({ input }) => input);
      const outputs = trainingSubset.map(({ output }) => output);
      const trainingSubsetCost = network.train(inputs, outputs);
      console.log(trainingSubsetCost);
      averageCost += trainingSubsetCost;
    }
    costEl.innerText = format(averageCost * subsetLength / (j - 1)).toString();
  }
});

let testIndex;
const context = canvasEl.getContext('2d');
numberEl.addEventListener('change', (event: MouseEvent) => {
  const target = event.target as HTMLSelectElement;
  if (target.value !== '') {
    testIndex = Number(target.value);
    guessEl.removeAttribute('disabled');
    const input = test[testIndex].input;
    const data = new Uint8ClampedArray(28 * 28 * 4);
    for (let i = 0; i < input.length; i++) {
      const index = i * 4;
      const value = input[i] * 255;
      data[index] = value;
      data[index + 1] = value;
      data[index + 2] = value;
      data[index + 3] = 255;
    }
    const imageData = new ImageData(data, 28, 28);
    context.putImageData(imageData, 0, 0);
  } else {
    guessEl.setAttribute('disabled', '');
  }
})

guessEl.addEventListener('click', () => {
  network.guess(test[testIndex].input);
  const results = network.getActivations();
  const max = Math.max(...results);
  const resultIndex = results.indexOf(max);
  resultEl.innerText = outputLabels[resultIndex];
});

exportEl.addEventListener('click', () => {
  const element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(JSON.stringify(network.export())));
  element.setAttribute('download', 'model.json');
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
});

importEl.addEventListener('click', () => {
  const reader = new FileReader();
  reader.onload = () => {
    const model = JSON.parse(reader.result as string);
    network.import(model);
  };
  reader.readAsText(fileEl.files[0]);
});
