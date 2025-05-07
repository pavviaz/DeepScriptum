
const markdownContent = process.argv[2];

if (markdownContent === undefined) {
    process.exit(2);
}

var md = require('markdown-it')();
var mk;

try {
    const katexPlugin = require('@vscode/markdown-it-katex');
    mk = katexPlugin.default || katexPlugin;
} catch (importError) {
    process.exit(3);
}

try {
    md.use(mk, { "throwOnError": true });
} catch (pluginError) {
    process.exit(4);
}

try {
    md.render(markdownContent);
} catch (renderError) {
    process.exit(1);
}