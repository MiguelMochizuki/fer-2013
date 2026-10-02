/**
 * Every visible string of the page, in pt-BR. Adding another language means
 * adding a table with the same keys. No dashes, no filler.
 */
import { formatBytes } from "./view.js";

export const S = {
  brand: "FER-2013",
  title: "Que emoção o modelo vê num rosto?",
  lede: "Envie uma foto. O modelo acha os rostos, classifica a expressão de cada um e mostra, num mapa de calor, para onde olhou.",
  navCode: "Código",
  navApi: "API",
  choosePhoto: "Escolher foto",
  tryExample: "Usar a foto de exemplo",
  dropHint: "Ou solte uma foto em qualquer lugar da página.",
  plateCaption: "Foto de exemplo",
  formats: "JPEG, PNG ou WebP, até 10\u00a0MB.",
  privacy: "A foto é processada neste navegador e não é enviada a ninguém.",
  loadingModel: (loaded, total) => `Baixando o modelo, ${formatBytes(loaded)} de ${formatBytes(total)}`,
  detecting: "Detectando rostos",
  result: (n, ms) => (n === 1 ? `1 rosto encontrado em ${Math.round(ms)} ms.` : `${n} rostos encontrados em ${Math.round(ms)} ms.`),
  noFace: "Nenhum rosto encontrado.",
  noFaceTips: "Tente uma foto mais nítida, com o rosto de frente e bem iluminado.",
  faceLabel: (i) => `Rosto ${i}`,
  original: "Original",
  heatmap: "Heatmap",
  heatmapAlt: (i, emotion) => `Mapa de calor do rosto ${i}, previsão ${emotion}`,
  stageLabel: (n) => (n === 0 ? "Foto enviada, nenhum rosto encontrado" : n === 1 ? "Foto enviada com 1 rosto marcado" : `Foto enviada com ${n} rostos marcados`),
  timings: { detect: "Detecção", classify: "Classificação", gradcam: "Grad-CAM++", runtime: "Execução" },
  backend: "WebAssembly, 1 thread",
  limitation: "Rótulos de emoção a partir de rostos não são confiáveis. Cerca de 71% de acurácia no FER-2013.",
  limitationLink: "Limitações",
  retry: "Tentar de novo",
  emotions: { angry: "raiva", disgust: "nojo", fear: "medo", happy: "alegria", sad: "tristeza", surprise: "surpresa", neutral: "neutro" },
  errors: {
    format: "Esse formato não é aceito. Use JPEG, PNG ou WebP.",
    size: "O arquivo passa de 10 MB.",
    pixels: "A imagem passa de 25 megapixels.",
    integrity: "O modelo baixado não confere com o esperado. Tente de novo.",
    download: "Não foi possível baixar o modelo. Verifique a conexão e tente de novo.",
    decode: "Não foi possível abrir essa imagem.",
    inference: "Algo deu errado ao analisar a foto.",
    unsupported: "Este navegador não tem WebAssembly com SIMD, que o modelo precisa.",
  },
};
