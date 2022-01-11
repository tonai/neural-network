import { create, addDependencies, mapDependencies, multiplyDependencies, transposeDependencies } from 'mathjs';

const math = create({ addDependencies, mapDependencies, multiplyDependencies, transposeDependencies }, {});

export const { add, map, multiply, transpose } = math;
