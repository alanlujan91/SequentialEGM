import{a as l}from"/SequentialEGM/build/_shared/chunk-3MZURCDM.js";import{a as o}from"/SequentialEGM/build/_shared/chunk-5JQ5LNKE.js";import{c as i}from"/SequentialEGM/build/_shared/chunk-2NH4LW52.js";var b=i((d,t)=>{var g=o(),s=l();t.exports=n;n.displayName="erb";n.aliases=[];function n(r){r.register(g),r.register(s),function(e){e.languages.erb={delimiter:{pattern:/^(\s*)<%=?|%>(?=\s*$)/,lookbehind:!0,alias:"punctuation"},ruby:{pattern:/\s*\S[\s\S]*/,alias:"language-ruby",inside:e.languages.ruby}},e.hooks.add("before-tokenize",function(a){var u=/<%=?(?:[^\r\n]|[\r\n](?!=begin)|[\r\n]=begin\s(?:[^\r\n]|[\r\n](?!=end))*[\r\n]=end)+?%>/g;e.languages["markup-templating"].buildPlaceholders(a,"erb",u)}),e.hooks.add("after-tokenize",function(a){e.languages["markup-templating"].tokenizePlaceholders(a,"erb")})}(r)}});export{b as a};
