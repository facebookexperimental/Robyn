
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';
export default [
{
  path: '/',
  component: ComponentCreator('/','deb'),
  exact: true,
},
{
  path: '/__docusaurus/debug',
  component: ComponentCreator('/__docusaurus/debug','fdd'),
  exact: true,
},
{
  path: '/blog',
  component: ComponentCreator('/blog','728'),
  exact: true,
},
{
  path: '/blog/hello-world',
  component: ComponentCreator('/blog/hello-world','798'),
  exact: true,
},
{
  path: '/blog/hola',
  component: ComponentCreator('/blog/hola','08d'),
  exact: true,
},
{
  path: '/blog/tags',
  component: ComponentCreator('/blog/tags','3a8'),
  exact: true,
},
{
  path: '/blog/tags/docusaurus',
  component: ComponentCreator('/blog/tags/docusaurus','917'),
  exact: true,
},
{
  path: '/blog/tags/facebook',
  component: ComponentCreator('/blog/tags/facebook','c2b'),
  exact: true,
},
{
  path: '/blog/tags/hello',
  component: ComponentCreator('/blog/tags/hello','c12'),
  exact: true,
},
{
  path: '/blog/tags/hola',
  component: ComponentCreator('/blog/tags/hola','084'),
  exact: true,
},
{
  path: '/blog/welcome',
  component: ComponentCreator('/blog/welcome','cb0'),
  exact: true,
},
{
  path: '/docs',
  component: ComponentCreator('/docs','19a'),
  
  routes: [
{
  path: '/docs/',
  component: ComponentCreator('/docs/','f0b'),
  exact: true,
},
{
  path: '/docs/doc1',
  component: ComponentCreator('/docs/doc1','392'),
  exact: true,
},
{
  path: '/docs/doc10',
  component: ComponentCreator('/docs/doc10','120'),
  exact: true,
},
{
  path: '/docs/doc3',
  component: ComponentCreator('/docs/doc3','e02'),
  exact: true,
},
{
  path: '/docs/doc4',
  component: ComponentCreator('/docs/doc4','50f'),
  exact: true,
},
{
  path: '/docs/doc5',
  component: ComponentCreator('/docs/doc5','b2b'),
  exact: true,
},
{
  path: '/docs/doc6',
  component: ComponentCreator('/docs/doc6','b6d'),
  exact: true,
},
{
  path: '/docs/doc7',
  component: ComponentCreator('/docs/doc7','ad0'),
  exact: true,
},
{
  path: '/docs/doc8',
  component: ComponentCreator('/docs/doc8','728'),
  exact: true,
},
{
  path: '/docs/doc9',
  component: ComponentCreator('/docs/doc9','1a8'),
  exact: true,
},
{
  path: '/docs/mdx',
  component: ComponentCreator('/docs/mdx','955'),
  exact: true,
},
]
},
{
  path: '*',
  component: ComponentCreator('*')
}
];
