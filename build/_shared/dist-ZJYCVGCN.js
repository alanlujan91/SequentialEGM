import"/SequentialEGM/build/_shared/chunk-RAQ24GF6.js";var n={name:"grid",arg:{type:String},body:{type:"myst",required:!0},run(r){return[{type:"grid",columns:i(r.arg),children:r.body}]}};function i(r){let t=(r??"1 2 2 3").split(/\s/).map(e=>Number(e.trim())).filter(e=>!Number.isNaN(e)).map(e=>Math.min(Math.max(Math.floor(e),1),12));return t.length===0||t.length>4?[1,2,2,3]:t}export{n as gridDirective};
