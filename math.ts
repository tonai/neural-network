import { create, addDependencies, mapDependencies, multiplyDependencies } from 'mathjs';

const math = create({ addDependencies, mapDependencies, multiplyDependencies }, {});

export const { add, map, multiply } = math;
