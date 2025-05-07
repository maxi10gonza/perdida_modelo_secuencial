let modelo;
let historial = [];

async function entrenarModelo() {
  const xs = tf.tensor1d([1, 2, 3, 4]);
  const ys = tf.tensor1d([3, 5, 7, 9]);

  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  modelo.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  const perdidas = [];
  await modelo.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        perdidas.push({ epoch: epoch, loss: logs.loss });
      }
    }
  });

  historial = perdidas;
  graficarPerdida(perdidas);

  document.getElementById("respuesta").style.display = "none";
}

function graficarPerdida(data) {
  const epocas = data.map(d => d.epoch);
  const perdidas = data.map(d => d.loss);

  const trace = {
    x: epocas,
    y: perdidas,
    mode: 'lines+markers',
    name: 'Pérdida (Loss)',
    line: { color: 'teal' },
    marker: { size: 6 }
  };

  const layout = {
    xaxis: { title: 'Época' },
    yaxis: { title: 'Valor de Pérdida' },
    legend: { x: 0.5, xanchor: 'center' },
    margin: { t: 20 }
  };

  Plotly.newPlot('plot', [trace], layout);

  const inicial = historial[0].loss.toFixed(4);
  const final = historial[historial.length - 1].loss.toFixed(4);
  const reduccion = ((1 - final / inicial) * 100).toFixed(2);
  document.getElementById("perdidaFinal").innerText =
    `Pérdida inicial: ${inicial}, Pérdida final: ${final} (Reducción: ${reduccion}%)`;
}

async function predecir() {
  if (!modelo) return alert("Entrena el modelo primero");

  const entrada = document.getElementById("entradaX").value;
  const valores = entrada.split(',').map(n => parseFloat(n.trim())).filter(n => !isNaN(n));

  const predicciones = await modelo.predict(tf.tensor1d(valores)).array();
  const resultados = predicciones.map(r => r[0]);

  let html = "<ul>";
  valores.forEach((x, i) => {
    html += `<li>• Para x = ${x}: y = ${resultados[i].toFixed(2)}</li>`;
  });
  html += "</ul>";

  document.getElementById("estado").innerText = "Estado: Modelo entrenado correctamente";
  document.getElementById("resultados").innerHTML = html;
  document.getElementById("respuesta").style.display = "block";
}
