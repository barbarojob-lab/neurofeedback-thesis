// EEGSimulator test script - Node.js puro, sin browser
const fs = require('fs');
try {
  const EEGSimulator = require('./dist/eeg-simulator.js').EEGSimulator;
  console.log('✅ EEGSimulator importado correctamente desde dist/');
  
  const sim = new EEGSimulator();
  sim.on('sample', (sample) => {
    console.log(`Sample: ch0=${sample.channelData[0].toFixed(1)}μV`);
    process.exit(0); // salir después de 1 sample
  });
  sim.start();
  console.log('🚀 Simulador iniciado - esperando 1 sample...');
} catch (error) {
  console.error('❌ Error:', error.message);
  console.log('Verificando archivos:');
  console.log('- backend/dist/eeg-simulator.js existe?', fs.existsSync('./dist/eeg-simulator.js'));
  console.log('- backend/dist/eeg-simulator.d.ts existe?', fs.existsSync('./dist/eeg-simulator.d.ts'));
  process.exit(1);
}
