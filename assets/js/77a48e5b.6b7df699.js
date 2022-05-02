"use strict";(self.webpackChunkmmm_for_all=self.webpackChunkmmm_for_all||[]).push([[277],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return m}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),p=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(s.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),d=p(n),m=a,f=d["".concat(s,".").concat(m)]||d[m]||c[m]||i;return n?r.createElement(f,o(o({ref:t},u),{},{components:n})):r.createElement(f,o({ref:t},u))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,o=new Array(i);o[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,o[1]=l;for(var p=2;p<i;p++)o[p]=n[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},3919:function(e,t,n){function r(e){return!0===/^(\w*:|\/\/)/.test(e)}function a(e){return void 0!==e&&!r(e)}n.d(t,{b:function(){return r},Z:function(){return a}})},4996:function(e,t,n){n.d(t,{C:function(){return i},Z:function(){return o}});var r=n(2263),a=n(3919);function i(){var e=(0,r.Z)().siteConfig,t=(e=void 0===e?{}:e).baseUrl,n=void 0===t?"/":t,i=e.url;return{withBaseUrl:function(e,t){return function(e,t,n,r){var i=void 0===r?{}:r,o=i.forcePrependBaseUrl,l=void 0!==o&&o,s=i.absolute,p=void 0!==s&&s;if(!n)return n;if(n.startsWith("#"))return n;if((0,a.b)(n))return n;if(l)return t+n;var u=n.startsWith(t)?n:t+n.replace(/^\//,"");return p?e+u:u}(i,n,e,t)}}}function o(e,t){return void 0===t&&(t={}),(0,i().withBaseUrl)(e,t)}},6197:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return p},toc:function(){return u},default:function(){return d}});var r=n(7462),a=n(3366),i=(n(7294),n(3905)),o=(n(4996),["components"]),l={id:"releases",title:"Releases"},s=void 0,p={unversionedId:"releases",id:"releases",isDocsHomePage:!1,title:"Releases",description:"3.6.2",source:"@site/docs/releases.mdx",sourceDirName:".",slug:"/releases",permalink:"/Robyn/docs/releases",editUrl:"https://github.com/facebookexperimental/Robyn/edit/main/website/docs/releases.mdx",tags:[],version:"current",frontMatter:{id:"releases",title:"Releases"},sidebar:"someSidebar",previous:{title:"Features",permalink:"/Robyn/docs/features"},next:{title:"Success Stories using Robyn",permalink:"/Robyn/docs/success-stories"}},u=[{value:"3.6.2",id:"362",children:[]},{value:"3.6.0",id:"360",children:[]}],c={toc:u};function d(e){var t=e.components,n=(0,a.Z)(e,o);return(0,i.kt)("wrapper",(0,r.Z)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"362"},"3.6.2"),(0,i.kt)("p",null,"31-3-2022\nAllocation and plot improvements, new warnings, bugs fixed"),(0,i.kt)("p",null,"Relevant changes on v3.6.2:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Viz"),": removed redundant information on plots and standardized styles and contents on all visualizations."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Feat"),": new ",(0,i.kt)("inlineCode",{parentName:"li"},"date_min")," and ",(0,i.kt)("inlineCode",{parentName:"li"},"date_max")," parameters on ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_allocator()")," to pick specific date range to consider mean spend values (",(0,i.kt)("a",{parentName:"li",href:"https://www.facebook.com/groups/robynmmm/permalink/1072870463481086"},"user request"),")."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Feat"),": new ",(0,i.kt)("inlineCode",{parentName:"li"},"plot")," methods for ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_allocator()")," and ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_save()")," outputs, and ",(0,i.kt)("inlineCode",{parentName:"li"},"print")," method for ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_inputs()")," with and without hyperparameters."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Feat"),": provide recommendations on calibration inputs depending on the experiments' confidence, spending, and KPI measured (#307)."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Feat"),': warn and avoid weekly trend input when data granularity is larger than "week".'),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Fix"),": issues on several ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_allocator()")," specific cases (#349, #344, #345), especially when some coefficients were 0."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Fix"),": bug with Weibull adstock scenario (#353)."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"Docs"),": fixed some typos, updated, and standardized internal documentation. ")),(0,i.kt)("hr",null),(0,i.kt)("h2",{id:"360"},"3.6.0"),(0,i.kt)("p",null,"22-2-2022"),(0,i.kt)("p",null,"What's New in v3.6.0"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"New ",(0,i.kt)("strong",{parentName:"li"},'hyperparameter "lambda"')," finds MOO-optimal lambda and thus removes the need of manual lambda selection."),(0,i.kt)("li",{parentName:"ul"},"New optional ",(0,i.kt)("strong",{parentName:"li"},"hyperparameter ",(0,i.kt)("inlineCode",{parentName:"strong"},"penalty.factor"))," that further extends hyperparameter spaces and thus potentially better fit."),(0,i.kt)("li",{parentName:"ul"},"New ",(0,i.kt)("strong",{parentName:"li"},"optimisation convergence rules & plots")," for each objective function showing if set iterations have converged or not (NRMSE, DECOMP.RSSD, and MAPE if calibrated)"),(0,i.kt)("li",{parentName:"ul"},"Improved ",(0,i.kt)("strong",{parentName:"li"},"response function")," now also returns the response for exposure metrics (response on imps, GRP, newsletter sendings, etc) and plots. Note that argument names and output class has changed. See updated ",(0,i.kt)("inlineCode",{parentName:"li"},"demo.R")," for more details."),(0,i.kt)("li",{parentName:"ul"},"More ",(0,i.kt)("strong",{parentName:"li"},"budget allocation stability")," by defaulting fitting media variables from ",(0,i.kt)("inlineCode",{parentName:"li"},"paid_media_vars")," to ",(0,i.kt)("inlineCode",{parentName:"li"},"paid_media_spends"),". Spend exposure fitting with Michaelis Menten function will only serve ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_response()")," function output and plotting. ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_allocator()")," now only relies on direct spend - response transformation."),(0,i.kt)("li",{parentName:"ul"},"Default ",(0,i.kt)("strong",{parentName:"li"},"beta coefficient signs"),": positive for paid & organic media and unconstrained for the rest. Users can still set signs manually."),(0,i.kt)("li",{parentName:"ul"},"New ",(0,i.kt)("strong",{parentName:"li"},"print methods")," for ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_inputs()"),", ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_run()"),", ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_outputs()"),", and ",(0,i.kt)("inlineCode",{parentName:"li"},"robyn_allocator()")," outputs to enable visibility on each step's results and objects content.")),(0,i.kt)("hr",null))}d.isMDXComponent=!0}}]);