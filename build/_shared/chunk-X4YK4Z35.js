import{A as H,B as y,E as x,F as q,Ka as oe,T as se,a as B,c as g,d as j,e as T,f as E,i as u,j as $,k as A,l as W,la as w,n as F,o as b,p as N,pa as K,q as P,t as U,u as V,x as ne}from"/SequentialEGM/build/_shared/chunk-ESBRTUPV.js";import{b as ie}from"/SequentialEGM/build/_shared/chunk-2NH4LW52.js";function Q(i){let e=Object.keys(i).join(""),t=/\w/.test(e);return t&&(e=e.replace(/\w/g,"")),`[${t?"\\w":""}${e.replace(/[^\w\s]/g,"\\$&")}]`}function le(i){let e=Object.create(null),t=Object.create(null);for(let{label:s}of i){e[s[0]]=!0;for(let o=1;o<s.length;o++)t[s[o]]=!0}let n=Q(e)+Q(t)+"*$";return[new RegExp("^"+n),new RegExp(n)]}function Fe(i){let e=i.map(s=>typeof s=="string"?{label:s}:s),[t,n]=e.every(s=>/^\w+$/.test(s.label))?[/\w*$/,/\w+$/]:le(e);return s=>{let o=s.matchBefore(n);return o||s.explicit?{from:o?o.from:s.pos,options:e,validFor:t}:null}}function Ne(i,e){return t=>{for(let n=w(t.state).resolveInner(t.pos,-1);n;n=n.parent){if(i.indexOf(n.name)>-1)return null;if(n.type.isTop)break}return e(t)}}function re(i,e){var t;let{source:n}=i,s=e&&n[0]!="^",o=n[n.length-1]!="$";return!s&&!o?i:new RegExp(`${s?"^":""}(?:${n})${o?"$":""}`,(t=i.flags)!==null&&t!==void 0?t:i.ignoreCase?"i":"")}function k(i,e){return u.create(i.filter(t=>t.field==e).map(t=>u.range(t.from,t.to)))}function pe(i){let e=C.parse(i);return(t,n,s,o)=>{let{text:r,ranges:a}=e.instantiate(t.state,s),l={changes:{from:s,to:o,insert:B.of(r)},scrollIntoView:!0,annotations:n?[ae.of(n),N.userEvent.of("input.complete")]:void 0};if(a.length&&(l.selection=k(a,0)),a.some(f=>f.field>0)){let f=new p(a,0),c=l.effects=[O.of(f)];t.state.field(I,!1)===void 0&&c.push(b.appendConfig.of([I,ye,xe,ce]))}t.dispatch(t.state.update(l))}}function J(i){return({state:e,dispatch:t})=>{let n=e.field(I,!1);if(!n||i<0&&n.active==0)return!1;let s=n.active+i,o=i>0&&!n.ranges.some(r=>r.field==s+i);return t(e.update({selection:k(n.ranges,s),effects:O.of(o?null:new p(n.ranges,s)),scrollIntoView:!0})),!0}}function Ve(i,e){return Object.assign(Object.assign({},e),{apply:pe(i)})}function He(){return[ve,Z]}function _(i){for(let e=0;e<R.length;e+=2)if(R.charCodeAt(e)==i)return R.charAt(e+1);return j(i<128?i:i+1)}function ee(i,e){return i.languageDataAt("closeBrackets",e)[0]||S}function Ie(i,e){let t=ee(i,i.selection.main.head),n=t.brackets||S.brackets;for(let s of n){let o=_(g(s,0));if(e==s)return o==s?Ee(i,s,n.indexOf(s+s+s)>-1,t):Oe(i,s,o,t.before||S.before);if(e==o&&te(i,i.selection.main.from))return Te(i,s,o)}return null}function te(i,e){let t=!1;return i.field(Z).between(0,i.doc.length,n=>{n==e&&(t=!0)}),t}function D(i,e){let t=i.sliceString(e,e+2);return t.slice(0,T(g(t,0)))}function Se(i,e){let t=i.sliceString(e-2,e);return T(g(t,0))==t.length?t:t.slice(1)}function Oe(i,e,t,n){let s=null,o=i.changeByRange(r=>{if(!r.empty)return{changes:[{insert:e,from:r.from},{insert:t,from:r.to}],effects:d.of(r.to+e.length),range:u.range(r.anchor+e.length,r.head+e.length)};let a=D(i.doc,r.head);return!a||/\s/.test(a)||n.indexOf(a)>-1?{changes:{insert:e+t,from:r.head},effects:d.of(r.head+e.length),range:u.cursor(r.head+e.length)}:{range:s=r}});return s?null:i.update(o,{scrollIntoView:!0,userEvent:"input.type"})}function Te(i,e,t){let n=null,s=i.changeByRange(o=>o.empty&&D(i.doc,o.head)==t?{changes:{from:o.head,to:o.head+t.length,insert:t},range:u.cursor(o.head+t.length)}:n={range:o});return n?null:i.update(s,{scrollIntoView:!0,userEvent:"input.type"})}function Ee(i,e,t,n){let s=n.stringPrefixes||S.stringPrefixes,o=null,r=i.changeByRange(a=>{if(!a.empty)return{changes:[{insert:e,from:a.from},{insert:e,from:a.to}],effects:d.of(a.to+e.length),range:u.range(a.anchor+e.length,a.head+e.length)};let l=a.head,f=D(i.doc,l),c;if(f==e){if(Y(i,l))return{changes:{insert:e+e,from:l},effects:d.of(l+e.length),range:u.cursor(l+e.length)};if(te(i,l)){let m=t&&i.sliceDoc(l,l+e.length*3)==e+e+e?e+e+e:e;return{changes:{from:l,to:l+m.length,insert:m},range:u.cursor(l+m.length)}}}else{if(t&&i.sliceDoc(l-2*e.length,l)==e+e&&(c=G(i,l-2*e.length,s))>-1&&Y(i,c))return{changes:{insert:e+e+e+e,from:l},effects:d.of(l+e.length),range:u.cursor(l+e.length)};if(i.charCategorizer(l)(f)!=P.Word&&G(i,l,s)>-1&&!Pe(i,l,e,s))return{changes:{insert:e+e,from:l},effects:d.of(l+e.length),range:u.cursor(l+e.length)}}return{range:o=a}});return o?null:i.update(r,{scrollIntoView:!0,userEvent:"input.type"})}function Y(i,e){let t=w(i).resolveInner(e+1);return t.parent&&t.from==e}function Pe(i,e,t,n){let s=w(i).resolveInner(e,-1),o=n.reduce((r,a)=>Math.max(r,a.length),0);for(let r=0;r<5;r++){let a=i.sliceDoc(s.from,Math.min(s.to,s.from+t.length+o)),l=a.indexOf(t);if(!l||l>-1&&n.indexOf(a.slice(0,l))>-1){let c=s.firstChild;for(;c&&c.from==s.from&&c.to-c.from>t.length+l;){if(i.sliceDoc(c.to-t.length,c.to)==t)return!1;c=c.firstChild}return!0}let f=s.to==e&&s.parent;if(!f)break;s=f}return!1}function G(i,e,t){let n=i.charCategorizer(e);if(n(i.sliceDoc(e-1,e))!=P.Word)return e;for(let s of t){let o=e-s.length;if(i.sliceDoc(o,e)==s&&n(i.sliceDoc(o-1,o))!=P.Word)return o}return-1}var z,ae,Ue,ce,L,v,C,fe,he,p,O,ue,I,de,me,ge,be,X,ye,xe,S,d,M,Z,R,we,ve,Ce,qe,De=ie(()=>{ne();se();oe();z=class{constructor(e,t,n,s){this.state=e,this.pos=t,this.explicit=n,this.view=s,this.abortListeners=[],this.abortOnDocChange=!1}tokenBefore(e){let t=w(this.state).resolveInner(this.pos,-1);for(;t&&e.indexOf(t.name)<0;)t=t.parent;return t?{from:t.from,to:this.pos,text:this.state.sliceDoc(t.from,this.pos),type:t.type}:null}matchBefore(e){let t=this.state.doc.lineAt(this.pos),n=Math.max(t.from,this.pos-250),s=t.text.slice(n-t.from,this.pos-t.from),o=s.search(re(e,!1));return o<0?null:{from:n+o,to:this.pos,text:s.slice(o)}}get aborted(){return this.abortListeners==null}addEventListener(e,t,n){e=="abort"&&this.abortListeners&&(this.abortListeners.push(t),n&&n.onDocChange&&(this.abortOnDocChange=!0))}};ae=F.define(),Ue=typeof navigator=="object"&&/Win/.test(navigator.platform),ce=x.baseTheme({".cm-tooltip.cm-tooltip-autocomplete":{"& > ul":{fontFamily:"monospace",whiteSpace:"nowrap",overflow:"hidden auto",maxWidth_fallback:"700px",maxWidth:"min(700px, 95vw)",minWidth:"250px",maxHeight:"10em",height:"100%",listStyle:"none",margin:0,padding:0,"& > li, & > completion-section":{padding:"1px 3px",lineHeight:1.2},"& > li":{overflowX:"hidden",textOverflow:"ellipsis",cursor:"pointer"},"& > completion-section":{display:"list-item",borderBottom:"1px solid silver",paddingLeft:"0.5em",opacity:.7}}},"&light .cm-tooltip-autocomplete ul li[aria-selected]":{background:"#17c",color:"white"},"&light .cm-tooltip-autocomplete-disabled ul li[aria-selected]":{background:"#777"},"&dark .cm-tooltip-autocomplete ul li[aria-selected]":{background:"#347",color:"white"},"&dark .cm-tooltip-autocomplete-disabled ul li[aria-selected]":{background:"#444"},".cm-completionListIncompleteTop:before, .cm-completionListIncompleteBottom:after":{content:'"\xB7\xB7\xB7"',opacity:.5,display:"block",textAlign:"center"},".cm-tooltip.cm-completionInfo":{position:"absolute",padding:"3px 9px",width:"max-content",maxWidth:"400px",boxSizing:"border-box",whiteSpace:"pre-line"},".cm-completionInfo.cm-completionInfo-left":{right:"100%"},".cm-completionInfo.cm-completionInfo-right":{left:"100%"},".cm-completionInfo.cm-completionInfo-left-narrow":{right:"30px"},".cm-completionInfo.cm-completionInfo-right-narrow":{left:"30px"},"&light .cm-snippetField":{backgroundColor:"#00000022"},"&dark .cm-snippetField":{backgroundColor:"#ffffff22"},".cm-snippetFieldPosition":{verticalAlign:"text-top",width:0,height:"1.15em",display:"inline-block",margin:"0 -0.7px -.7em",borderLeft:"1.4px dotted #888"},".cm-completionMatchedText":{textDecoration:"underline"},".cm-completionDetail":{marginLeft:"0.5em",fontStyle:"italic"},".cm-completionIcon":{fontSize:"90%",width:".8em",display:"inline-block",textAlign:"center",paddingRight:".6em",opacity:"0.6",boxSizing:"content-box"},".cm-completionIcon-function, .cm-completionIcon-method":{"&:after":{content:"'\u0192'"}},".cm-completionIcon-class":{"&:after":{content:"'\u25CB'"}},".cm-completionIcon-interface":{"&:after":{content:"'\u25CC'"}},".cm-completionIcon-variable":{"&:after":{content:"'\u{1D465}'"}},".cm-completionIcon-constant":{"&:after":{content:"'\u{1D436}'"}},".cm-completionIcon-type":{"&:after":{content:"'\u{1D461}'"}},".cm-completionIcon-enum":{"&:after":{content:"'\u222A'"}},".cm-completionIcon-property":{"&:after":{content:"'\u25A1'"}},".cm-completionIcon-keyword":{"&:after":{content:"'\u{1F511}\uFE0E'"}},".cm-completionIcon-namespace":{"&:after":{content:"'\u25A2'"}},".cm-completionIcon-text":{"&:after":{content:"'abc'",fontSize:"50%",verticalAlign:"middle"}}}),L=class{constructor(e,t,n,s){this.field=e,this.line=t,this.from=n,this.to=s}},v=class{constructor(e,t,n){this.field=e,this.from=t,this.to=n}map(e){let t=e.mapPos(this.from,-1,E.TrackDel),n=e.mapPos(this.to,1,E.TrackDel);return t==null||n==null?null:new v(this.field,t,n)}},C=class{constructor(e,t){this.lines=e,this.fieldPositions=t}instantiate(e,t){let n=[],s=[t],o=e.doc.lineAt(t),r=/^\s*/.exec(o.text)[0];for(let l of this.lines){if(n.length){let f=r,c=/^\t*/.exec(l)[0].length;for(let h=0;h<c;h++)f+=e.facet(K);s.push(t+f.length-c),l=f+l.slice(c)}n.push(l),t+=l.length+1}let a=this.fieldPositions.map(l=>new v(l.field,s[l.line]+l.from,s[l.line]+l.to));return{text:n,ranges:a}}static parse(e){let t=[],n=[],s=[],o;for(let r of e.split(/\r\n?|\n/)){for(;o=/[#$]\{(?:(\d+)(?::([^}]*))?|((?:\\[{}]|[^}])*))\}/.exec(r);){let a=o[1]?+o[1]:null,l=o[2]||o[3]||"",f=-1,c=l.replace(/\\[{}]/g,h=>h[1]);for(let h=0;h<t.length;h++)(a!=null?t[h].seq==a:c&&t[h].name==c)&&(f=h);if(f<0){let h=0;for(;h<t.length&&(a==null||t[h].seq!=null&&t[h].seq<a);)h++;t.splice(h,0,{seq:a,name:c}),f=h;for(let m of s)m.field>=f&&m.field++}s.push(new L(f,n.length,o.index,o.index+c.length)),r=r.slice(0,o.index)+l+r.slice(o.index+o[0].length)}r=r.replace(/\\([{}])/g,(a,l,f)=>{for(let c of s)c.line==n.length&&c.from>f&&(c.from--,c.to--);return l}),n.push(r)}return new C(n,s)}},fe=y.widget({widget:new class extends H{toDOM(){let i=document.createElement("span");return i.className="cm-snippetFieldPosition",i}ignoreEvent(){return!1}}}),he=y.mark({class:"cm-snippetField"}),p=class{constructor(e,t){this.ranges=e,this.active=t,this.deco=y.set(e.map(n=>(n.from==n.to?fe:he).range(n.from,n.to)))}map(e){let t=[];for(let n of this.ranges){let s=n.map(e);if(!s)return null;t.push(s)}return new p(t,this.active)}selectionInsideField(e){return e.ranges.every(t=>this.ranges.some(n=>n.field==this.active&&n.from<=t.from&&n.to>=t.to))}},O=b.define({map(i,e){return i&&i.map(e)}}),ue=b.define(),I=A.define({create(){return null},update(i,e){for(let t of e.effects){if(t.is(O))return t.value;if(t.is(ue)&&i)return new p(i.ranges,t.value)}return i&&e.docChanged&&(i=i.map(e.changes)),i&&e.selection&&!i.selectionInsideField(e.selection)&&(i=null),i},provide:i=>x.decorations.from(i,e=>e?e.deco:y.none)});de=({state:i,dispatch:e})=>i.field(I,!1)?(e(i.update({effects:O.of(null)})),!0):!1,me=J(1),ge=J(-1),be=[{key:"Tab",run:me,shift:ge},{key:"Escape",run:de}],X=$.define({combine(i){return i.length?i[0]:be}}),ye=W.highest(q.compute([X],i=>i.facet(X)));xe=x.domEventHandlers({mousedown(i,e){let t=e.state.field(I,!1),n;if(!t||(n=e.posAtCoords({x:i.clientX,y:i.clientY}))==null)return!1;let s=t.ranges.find(o=>o.from<=n&&o.to>=n);return!s||s.field==t.active?!1:(e.dispatch({selection:k(t.ranges,s.field),effects:O.of(t.ranges.some(o=>o.field>s.field)?new p(t.ranges,s.field):null),scrollIntoView:!0}),!0)}}),S={brackets:["(","[","{","'",'"'],before:")]}:;>",stringPrefixes:[]},d=b.define({map(i,e){let t=e.mapPos(i,-1,E.TrackAfter);return t??void 0}}),M=new class extends U{};M.startSide=1;M.endSide=-1;Z=A.define({create(){return V.empty},update(i,e){if(i=i.map(e.changes),e.selection){let t=e.state.doc.lineAt(e.selection.main.head);i=i.update({filter:n=>n>=t.from&&n<=t.to})}for(let t of e.effects)t.is(d)&&(i=i.update({add:[M.range(t.value,t.value+1)]}));return i}});R="()[]{}<>";we=typeof navigator=="object"&&/Android\b/.test(navigator.userAgent),ve=x.inputHandler.of((i,e,t,n)=>{if((we?i.composing:i.compositionStarted)||i.state.readOnly)return!1;let s=i.state.selection.main;if(n.length>2||n.length==2&&T(g(n,0))==1||e!=s.from||t!=s.to)return!1;let o=Ie(i.state,n);return o?(i.dispatch(o),!0):!1}),Ce=({state:i,dispatch:e})=>{if(i.readOnly)return!1;let n=ee(i,i.selection.main.head).brackets||S.brackets,s=null,o=i.changeByRange(r=>{if(r.empty){let a=Se(i.doc,r.head);for(let l of n)if(l==a&&D(i.doc,r.head)==_(g(l,0)))return{changes:{from:r.head-l.length,to:r.head+l.length},range:u.cursor(r.head-l.length)}}return{range:s=r}});return s||e(i.update(o,{scrollIntoView:!0,userEvent:"delete.backward"})),!s},qe=[{key:"Backspace",run:Ce}]});export{z as a,Fe as b,Ne as c,Ve as d,He as e,qe as f,De as g};
