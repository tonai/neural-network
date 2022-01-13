export function rand(): number {
  return random(-1, 1)
}

export function random(from: number, to: number): number {
  return Math.random() * (to - from) + from;
}

export function format(number: number): number {
  return Math.round(number * 100) / 100;
}

export function sigmoid(x: number): number {
  return 1 / ( 1 + Math.exp(-x) );
}

export function sigmoidPrime(x: number): number {
  return sigmoid(x) * ( 1 - sigmoid(x) );
}

export function convertToDecimal(hex: string): [number, number, number] {
  hex = hex.slice(1);
  const r = parseInt(hex.slice(0, 2), 16) / 255;
  const g = parseInt(hex.slice(2, 4), 16) / 255;
  const b = parseInt(hex.slice(4, 6), 16) / 255;
  return [r, g, b];
}

export function convertToHex(input: [number, number, number] ): string {
  const hex = [];
  hex[0] = Math.round(input[0] * 255).toString(16).padStart(2, '0');
  hex[1] = Math.round(input[1] * 255).toString(16).padStart(2, '0');
  hex[2] = Math.round(input[2] * 255).toString(16).padStart(2, '0');
  return hex.join('');
}

export function shuffle(array) {
  let currentIndex = array.length,  randomIndex;
  while (currentIndex != 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }
  return array;
}