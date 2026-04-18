import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://Shaswat2001.github.io',
  base: '/robosmith',
  integrations: [
    starlight({
      title: 'RoboSmith',
      description: 'Natural language to trained robot policies, plus tools for integrating existing robotics artifacts.',
      favicon: '/favicon.svg',
      logo: {
        light: './public/logo-light.svg',
        dark: './public/logo-dark.svg',
        replacesTitle: false,
      },
      customCss: ['./src/styles/custom.css'],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Shaswat2001/robosmith',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/Shaswat2001/robosmith/edit/main/robosmith-docs/',
      },
      sidebar: [
        {
          label: 'Start Here',
          items: [
            { label: 'The Problem', slug: '' },
            { label: 'Why RoboSmith?', slug: 'guides/why-robosmith' },
            { label: 'Installation', slug: 'guides/installation' },
            { label: 'Quick Start', slug: 'guides/quickstart' },
          ],
        },
        {
          label: 'Workflows',
          items: [
            { label: 'Training Pipeline', slug: 'workflows/training-pipeline' },
            { label: 'Integration Tooling', slug: 'workflows/integration-tooling' },
            { label: 'Runs And Resume', slug: 'workflows/runs' },
          ],
        },
        {
          label: 'Systems',
          items: [
            { label: 'Environments', slug: 'systems/environments' },
            { label: 'Reward Design', slug: 'systems/reward-design' },
            { label: 'Training Backends', slug: 'systems/training-backends' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'CLI Commands', slug: 'reference/cli' },
            { label: 'Python API', slug: 'reference/python-api' },
            { label: 'Configuration', slug: 'reference/configuration' },
            { label: 'Architecture', slug: 'reference/architecture' },
          ],
        },
        {
          label: 'Extending',
          items: [
            { label: 'Custom Trainers', slug: 'extending/trainers' },
            { label: 'Custom Environments', slug: 'extending/environments' },
          ],
        },
      ],
    }),
  ],
});
