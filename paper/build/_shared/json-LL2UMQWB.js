import{c as r}from"/SequentialEGM/build/_shared/chunk-RAQ24GF6.js";var O=r((_,o)=>{function l(n){let t={literal:"true false null"},i=[n.C_LINE_COMMENT_MODE,n.C_BLOCK_COMMENT_MODE],e=[n.QUOTE_STRING_MODE,n.C_NUMBER_MODE],E={end:",",endsWithParent:!0,excludeEnd:!0,contains:e,keywords:t},a={begin:/\{/,end:/\}/,contains:[{className:"attr",begin:/"/,end:/"/,contains:[n.BACKSLASH_ESCAPE],illegal:"\\n"},n.inherit(E,{begin:/:/})].concat(i),illegal:"\\S"},c={begin:"\\[",end:"\\]",contains:[n.inherit(E)],illegal:"\\S"};return e.push(a,c),i.forEach(function(s){e.push(s)}),{name:"JSON",contains:e,keywords:t,illegal:"\\S"}}o.exports=l});export default O();
