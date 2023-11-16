import{c}from"/SequentialEGM/build/_shared/chunk-RAQ24GF6.js";var m=c((h,o)=>{o.exports=i;i.displayName="factor";i.aliases=[];function i(l){(function(u){var r={function:/\b(?:BUGS?|FIX(?:MES?)?|NOTES?|TODOS?|XX+|HACKS?|WARN(?:ING)?|\?{2,}|!{2,})\b/},t={number:/\\[^\s']|%\w/},s={comment:[{pattern:/(^|\s)(?:! .*|!$)/,lookbehind:!0,inside:r},{pattern:/(^|\s)\/\*\s[\s\S]*?\*\/(?=\s|$)/,lookbehind:!0,greedy:!0,inside:r},{pattern:/(^|\s)!\[(={0,6})\[\s[\s\S]*?\]\2\](?=\s|$)/,lookbehind:!0,greedy:!0,inside:r}],number:[{pattern:/(^|\s)[+-]?\d+(?=\s|$)/,lookbehind:!0},{pattern:/(^|\s)[+-]?0(?:b[01]+|o[0-7]+|d\d+|x[\dA-F]+)(?=\s|$)/i,lookbehind:!0},{pattern:/(^|\s)[+-]?\d+\/\d+\.?(?=\s|$)/,lookbehind:!0},{pattern:/(^|\s)\+?\d+\+\d+\/\d+(?=\s|$)/,lookbehind:!0},{pattern:/(^|\s)-\d+-\d+\/\d+(?=\s|$)/,lookbehind:!0},{pattern:/(^|\s)[+-]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:e[+-]?\d+)?(?=\s|$)/i,lookbehind:!0},{pattern:/(^|\s)NAN:\s+[\da-fA-F]+(?=\s|$)/,lookbehind:!0},{pattern:/(^|\s)[+-]?0(?:b1\.[01]*|o1\.[0-7]*|d1\.\d*|x1\.[\dA-F]*)p\d+(?=\s|$)/i,lookbehind:!0}],regexp:{pattern:/(^|\s)R\/\s(?:\\\S|[^\\/])*\/(?:[idmsr]*|[idmsr]+-[idmsr]+)(?=\s|$)/,lookbehind:!0,alias:"number",inside:{variable:/\\\S/,keyword:/[+?*\[\]^$(){}.|]/,operator:{pattern:/(\/)[idmsr]+(?:-[idmsr]+)?/,lookbehind:!0}}},boolean:{pattern:/(^|\s)[tf](?=\s|$)/,lookbehind:!0},"custom-string":{pattern:/(^|\s)[A-Z0-9\-]+"\s(?:\\\S|[^"\\])*"/,lookbehind:!0,greedy:!0,alias:"string",inside:{number:/\\\S|%\w|\//}},"multiline-string":[{pattern:/(^|\s)STRING:\s+\S+(?:\n|\r\n).*(?:\n|\r\n)\s*;(?=\s|$)/,lookbehind:!0,greedy:!0,alias:"string",inside:{number:t.number,"semicolon-or-setlocal":{pattern:/([\r\n][ \t]*);(?=\s|$)/,lookbehind:!0,alias:"function"}}},{pattern:/(^|\s)HEREDOC:\s+\S+(?:\n|\r\n).*(?:\n|\r\n)\s*\S+(?=\s|$)/,lookbehind:!0,greedy:!0,alias:"string",inside:t},{pattern:/(^|\s)\[(={0,6})\[\s[\s\S]*?\]\2\](?=\s|$)/,lookbehind:!0,greedy:!0,alias:"string",inside:t}],"special-using":{pattern:/(^|\s)USING:(?:\s\S+)*(?=\s+;(?:\s|$))/,lookbehind:!0,alias:"function",inside:{string:{pattern:/(\s)[^:\s]+/,lookbehind:!0}}},"stack-effect-delimiter":[{pattern:/(^|\s)(?:call|eval|execute)?\((?=\s)/,lookbehind:!0,alias:"operator"},{pattern:/(\s)--(?=\s)/,lookbehind:!0,alias:"operator"},{pattern:/(\s)\)(?=\s|$)/,lookbehind:!0,alias:"operator"}],combinators:{pattern:null,lookbehind:!0,alias:"keyword"},"kernel-builtin":{pattern:null,lookbehind:!0,alias:"variable"},"sequences-builtin":{pattern:null,lookbehind:!0,alias:"variable"},"math-builtin":{pattern:null,lookbehind:!0,alias:"variable"},"constructor-word":{pattern:/(^|\s)<(?!=+>|-+>)\S+>(?=\s|$)/,lookbehind:!0,alias:"keyword"},"other-builtin-syntax":{pattern:null,lookbehind:!0,alias:"operator"},"conventionally-named-word":{pattern:/(^|\s)(?!")(?:(?:change|new|set|with)-\S+|\$\S+|>[^>\s]+|[^:>\s]+>|[^>\s]+>[^>\s]+|\+[^+\s]+\+|[^?\s]+\?|\?[^?\s]+|[^>\s]+>>|>>[^>\s]+|[^<\s]+<<|\([^()\s]+\)|[^!\s]+!|[^*\s]\S*\*|[^.\s]\S*\.)(?=\s|$)/,lookbehind:!0,alias:"keyword"},"colon-syntax":{pattern:/(^|\s)(?:[A-Z0-9\-]+#?)?:{1,2}\s+(?:;\S+|(?!;)\S+)(?=\s|$)/,lookbehind:!0,greedy:!0,alias:"function"},"semicolon-or-setlocal":{pattern:/(\s)(?:;|:>)(?=\s|$)/,lookbehind:!0,alias:"function"},"curly-brace-literal-delimiter":[{pattern:/(^|\s)[a-z]*\{(?=\s)/i,lookbehind:!0,alias:"operator"},{pattern:/(\s)\}(?=\s|$)/,lookbehind:!0,alias:"operator"}],"quotation-delimiter":[{pattern:/(^|\s)\[(?=\s)/,lookbehind:!0,alias:"operator"},{pattern:/(\s)\](?=\s|$)/,lookbehind:!0,alias:"operator"}],"normal-word":{pattern:/(^|\s)[^"\s]\S*(?=\s|$)/,lookbehind:!0},string:{pattern:/"(?:\\\S|[^"\\])*"/,greedy:!0,inside:t}},p=function(e){return(e+"").replace(/([.?*+\^$\[\]\\(){}|\-])/g,"\\$1")},n=function(e){return new RegExp("(^|\\s)(?:"+e.map(p).join("|")+")(?=\\s|$)")},a={"kernel-builtin":["or","2nipd","4drop","tuck","wrapper","nip","wrapper?","callstack>array","die","dupd","callstack","callstack?","3dup","hashcode","pick","4nip","build",">boolean","nipd","clone","5nip","eq?","?","=","swapd","2over","clear","2dup","get-retainstack","not","tuple?","dup","3nipd","call","-rotd","object","drop","assert=","assert?","-rot","execute","boa","get-callstack","curried?","3drop","pickd","overd","over","roll","3nip","swap","and","2nip","rotd","throw","(clone)","hashcode*","spin","reach","4dup","equal?","get-datastack","assert","2drop","<wrapper>","boolean?","identity-hashcode","identity-tuple?","null","composed?","new","5drop","rot","-roll","xor","identity-tuple","boolean"],"other-builtin-syntax":["=======","recursive","flushable",">>","<<<<<<","M\\","B","PRIVATE>","\\","======","final","inline","delimiter","deprecated","<PRIVATE",">>>>>>","<<<<<<<","parse-complex","malformed-complex","read-only",">>>>>>>","call-next-method","<<","foldable","$","$[","${"],"sequences-builtin":["member-eq?","mismatch","append","assert-sequence=","longer","repetition","clone-like","3sequence","assert-sequence?","last-index-from","reversed","index-from","cut*","pad-tail","join-as","remove-eq!","concat-as","but-last","snip","nths","nth","sequence","longest","slice?","<slice>","remove-nth","tail-slice","empty?","tail*","member?","virtual-sequence?","set-length","drop-prefix","iota","unclip","bounds-error?","unclip-last-slice","non-negative-integer-expected","non-negative-integer-expected?","midpoint@","longer?","?set-nth","?first","rest-slice","prepend-as","prepend","fourth","sift","subseq-start","new-sequence","?last","like","first4","1sequence","reverse","slice","virtual@","repetition?","set-last","index","4sequence","max-length","set-second","immutable-sequence","first2","first3","supremum","unclip-slice","suffix!","insert-nth","tail","3append","short","suffix","concat","flip","immutable?","reverse!","2sequence","sum","delete-all","indices","snip-slice","<iota>","check-slice","sequence?","head","append-as","halves","sequence=","collapse-slice","?second","slice-error?","product","bounds-check?","bounds-check","immutable","virtual-exemplar","harvest","remove","pad-head","last","set-fourth","cartesian-product","remove-eq","shorten","shorter","reversed?","shorter?","shortest","head-slice","pop*","tail-slice*","but-last-slice","iota?","append!","cut-slice","new-resizable","head-slice*","sequence-hashcode","pop","set-nth","?nth","second","join","immutable-sequence?","<reversed>","3append-as","virtual-sequence","subseq?","remove-nth!","length","last-index","lengthen","assert-sequence","copy","move","third","first","tail?","set-first","prefix","bounds-error","<repetition>","exchange","surround","cut","min-length","set-third","push-all","head?","subseq-start-from","delete-slice","rest","sum-lengths","head*","infimum","remove!","glue","slice-error","subseq","push","replace-slice","subseq-as","unclip-last"],"math-builtin":["number=","next-power-of-2","?1+","fp-special?","imaginary-part","float>bits","number?","fp-infinity?","bignum?","fp-snan?","denominator","gcd","*","+","fp-bitwise=","-","u>=","/",">=","bitand","power-of-2?","log2-expects-positive","neg?","<","log2",">","integer?","number","bits>double","2/","zero?","bits>float","float?","shift","ratio?","rect>","even?","ratio","fp-sign","bitnot",">fixnum","complex?","/i","integer>fixnum","/f","sgn",">bignum","next-float","u<","u>","mod","recip","rational",">float","2^","integer","fixnum?","neg","fixnum","sq","bignum",">rect","bit?","fp-qnan?","simple-gcd","complex","<fp-nan>","real",">fraction","double>bits","bitor","rem","fp-nan-payload","real-part","log2-expects-positive?","prev-float","align","unordered?","float","fp-nan?","abs","bitxor","integer>fixnum-strict","u<=","odd?","<=","/mod",">integer","real?","rational?","numerator"]};Object.keys(a).forEach(function(e){s[e].pattern=n(a[e])});var d=["2bi","while","2tri","bi*","4dip","both?","same?","tri@","curry","prepose","3bi","?if","tri*","2keep","3keep","curried","2keepd","when","2bi*","2tri*","4keep","bi@","keepdd","do","unless*","tri-curry","if*","loop","bi-curry*","when*","2bi@","2tri@","with","2with","either?","bi","until","3dip","3curry","tri-curry*","tri-curry@","bi-curry","keepd","compose","2dip","if","3tri","unless","tuple","keep","2curry","tri","most","while*","dip","composed","bi-curry@","find-last-from","trim-head-slice","map-as","each-from","none?","trim-tail","partition","if-empty","accumulate*","reject!","find-from","accumulate-as","collector-for-as","reject","map","map-sum","accumulate!","2each-from","follow","supremum-by","map!","unless-empty","collector","padding","reduce-index","replicate-as","infimum-by","trim-tail-slice","count","find-index","filter","accumulate*!","reject-as","map-integers","map-find","reduce","selector","interleave","2map","filter-as","binary-reduce","map-index-as","find","produce","filter!","replicate","cartesian-map","cartesian-each","find-index-from","map-find-last","3map-as","3map","find-last","selector-as","2map-as","2map-reduce","accumulate","each","each-index","accumulate*-as","when-empty","all?","collector-as","push-either","new-like","collector-for","2selector","push-if","2all?","map-reduce","3each","any?","trim-slice","2reduce","change-nth","produce-as","2each","trim","trim-head","cartesian-find","map-index","if-zero","each-integer","unless-zero","(find-integer)","when-zero","find-last-integer","(all-integers?)","times","(each-integer)","find-integer","all-integers?","unless-negative","if-positive","when-positive","when-negative","unless-positive","if-negative","case","2cleave","cond>quot","case>quot","3cleave","wrong-values","to-fixed-point","alist>quot","cond","cleave","call-effect","recursive-hashcode","spread","deep-spread>quot","2||","0||","n||","0&&","2&&","3||","1||","1&&","n&&","3&&","smart-unless*","keep-inputs","reduce-outputs","smart-when*","cleave>array","smart-with","smart-apply","smart-if","inputs/outputs","output>sequence-n","map-outputs","map-reduce-outputs","dropping","output>array","smart-map-reduce","smart-2map-reduce","output>array-n","nullary","input<sequence","append-outputs","drop-inputs","inputs","smart-2reduce","drop-outputs","smart-reduce","preserving","smart-when","outputs","append-outputs-as","smart-unless","smart-if*","sum-outputs","input<sequence-unsafe","output>sequence"];s.combinators.pattern=n(d),u.languages.factor=s})(l)}});export{m as a};
