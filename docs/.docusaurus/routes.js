
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';
export default [
{
  path: '/Robyn/',
  component: ComponentCreator('/Robyn/','0b2'),
  exact: true,
},
{
  path: '/Robyn/blog',
  component: ComponentCreator('/Robyn/blog','9e2'),
  exact: true,
},
{
  path: '/Robyn/blog/hello-world',
  component: ComponentCreator('/Robyn/blog/hello-world','368'),
  exact: true,
},
{
  path: '/Robyn/blog/hola',
  component: ComponentCreator('/Robyn/blog/hola','4bf'),
  exact: true,
},
{
  path: '/Robyn/blog/tags',
  component: ComponentCreator('/Robyn/blog/tags','925'),
  exact: true,
},
{
  path: '/Robyn/blog/tags/docusaurus',
  component: ComponentCreator('/Robyn/blog/tags/docusaurus','629'),
  exact: true,
},
{
  path: '/Robyn/blog/tags/facebook',
  component: ComponentCreator('/Robyn/blog/tags/facebook','5fb'),
  exact: true,
},
{
  path: '/Robyn/blog/tags/hello',
  component: ComponentCreator('/Robyn/blog/tags/hello','b4e'),
  exact: true,
},
{
  path: '/Robyn/blog/tags/hola',
  component: ComponentCreator('/Robyn/blog/tags/hola','bc9'),
  exact: true,
},
{
  path: '/Robyn/blog/welcome',
  component: ComponentCreator('/Robyn/blog/welcome','11f'),
  exact: true,
},
{
  path: '/Robyn/docs',
  component: ComponentCreator('/Robyn/docs','61a'),
  
  routes: [
{
  path: '/Robyn/docs/',
  component: ComponentCreator('/Robyn/docs/','345'),
  exact: true,
},
{
  path: '/Robyn/docs/doc1',
  component: ComponentCreator('/Robyn/docs/doc1','17f'),
  exact: true,
},
{
  path: '/Robyn/docs/doc10',
  component: ComponentCreator('/Robyn/docs/doc10','3fe'),
  exact: true,
},
{
  path: '/Robyn/docs/doc11',
  component: ComponentCreator('/Robyn/docs/doc11','7f3'),
  exact: true,
},
{
  path: '/Robyn/docs/doc12',
  component: ComponentCreator('/Robyn/docs/doc12','c8b'),
  exact: true,
},
{
  path: '/Robyn/docs/doc3',
  component: ComponentCreator('/Robyn/docs/doc3','005'),
  exact: true,
},
{
  path: '/Robyn/docs/doc4',
  component: ComponentCreator('/Robyn/docs/doc4','5fe'),
  exact: true,
},
{
  path: '/Robyn/docs/doc5',
  component: ComponentCreator('/Robyn/docs/doc5','eb4'),
  exact: true,
},
{
  path: '/Robyn/docs/doc6',
  component: ComponentCreator('/Robyn/docs/doc6','71f'),
  exact: true,
},
{
  path: '/Robyn/docs/doc7',
  component: ComponentCreator('/Robyn/docs/doc7','79c'),
  exact: true,
},
{
  path: '/Robyn/docs/doc8',
  component: ComponentCreator('/Robyn/docs/doc8','c77'),
  exact: true,
},
{
  path: '/Robyn/docs/doc9',
  component: ComponentCreator('/Robyn/docs/doc9','d37'),
  exact: true,
},
]
},
{
  path: '*',
  component: ComponentCreator('*')
}
];
