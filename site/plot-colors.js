"use strict";

const loopGradient = [
  "#301264", "#431378", "#56147e", "#6a187f",
  "#7c1d7b", "#8e2474", "#a42b68", "#b5335b",
  "#c63e4c", "#d34d41", "#dc5d37", "#e36e31",
  "#e97f29", "#ed9024", "#efa024", "#f0ad2a",
];

function loopColor(loop, passes) {
  const position = loop / Math.max(1, passes - 1);
  return loopGradient[Math.round(position * (loopGradient.length - 1))];
}
