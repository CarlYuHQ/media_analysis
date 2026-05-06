module.exports = {
  apps: [{
    name: 'media-report',
    script: 'npx',
    args: 'serve -s -l 3000',
    cwd: __dirname,
    env: {
      NODE_ENV: 'production'
    }
  }]
};
