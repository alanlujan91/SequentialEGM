import"/SequentialEGM/build/_shared/chunk-2NH4LW52.js";var i=/^(.+?)<([^<>]+)>$/,d={name:"button",doc:"Button element with an action to navigate to internal or external links.",body:{type:String,doc:"The body of the button.",required:!0},run(r){let o=r.body,t=i.exec(o),[,e,n]=t??[],l={type:"link",url:n??o,children:[],class:"button"};return e&&(l.children=[{type:"text",value:e.trim()}]),[l]}};export{d as buttonRole};
