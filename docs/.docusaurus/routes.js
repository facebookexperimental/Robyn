
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';
export default [
{
  path: '/Robyn/',
  component: ComponentCreator('/Robyn/','0b2'),
  exact: true,
},
{
  path: '/Robyn/docs',
  component: ComponentCreator('/Robyn/docs','701'),
  
  routes: [
{
  path: '/Robyn/docs/',
  component: ComponentCreator('/Robyn/docs/','982'),
  exact: true,
},
{
  path: '/Robyn/docs/about',
  component: ComponentCreator('/Robyn/docs/about','9e5'),
  exact: true,
},
{
  path: '/Robyn/docs/automated-hyperparameter-selection-optimization',
  component: ComponentCreator('/Robyn/docs/automated-hyperparameter-selection-optimization','49b'),
  exact: true,
},
{
  path: '/Robyn/docs/calibration',
  component: ComponentCreator('/Robyn/docs/calibration','e3c'),
  exact: true,
},
{
  path: '/Robyn/docs/contributing',
  component: ComponentCreator('/Robyn/docs/contributing','2c9'),
  exact: true,
},
{
  path: '/Robyn/docs/facebook-prophet',
  component: ComponentCreator('/Robyn/docs/facebook-prophet','607'),
  exact: true,
},
{
  path: '/Robyn/docs/outputs-diagnostics',
  component: ComponentCreator('/Robyn/docs/outputs-diagnostics','a83'),
  exact: true,
},
{
  path: '/Robyn/docs/quick-start',
  component: ComponentCreator('/Robyn/docs/quick-start','723'),
  exact: true,
},
{
  path: '/Robyn/docs/ridge-regression',
  component: ComponentCreator('/Robyn/docs/ridge-regression','8d0'),
  exact: true,
},
{
  path: '/Robyn/docs/step-by-step-guide',
  component: ComponentCreator('/Robyn/docs/step-by-step-guide','315'),
  exact: true,
},
{
  path: '/Robyn/docs/variable-transformations',
  component: ComponentCreator('/Robyn/docs/variable-transformations','67d'),
  exact: true,
},
]
},
{
  path: '*',
  component: ComponentCreator('*')
}
];
