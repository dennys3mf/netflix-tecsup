new autoComplete({
    data: {                              // Origen de datos [Array, Function, Async] | (OBLIGATORIO)
      src: films,
    },
    selector: "#autoComplete",           // Selector del campo de entrada              | (Opcional)
    threshold: 2,                        // Longitud mín. de caracteres para iniciar el motor | (Opcional)
    debounce: 100,                       // Duración posterior para que el motor arranque | (Opcional)
    searchEngine: "strict",              // Tipo/mode del motor de búsqueda           | (Opcional)
    resultsList: {                       // Objeto de lista de resultados renderizados   | (Opcional)
        render: true,
        container: source => {
            source.setAttribute("id", "food_list");
        },
        destination: document.querySelector("#autoComplete"),
        position: "afterend",
        element: "ul"
    },
    maxResults: 5,                         // Núm. máx. de resultados renderizados | (Opcional)
    highlight: true,                       // Resaltar resultados coincidentes      | (Opcional)
    resultItem: {                          // Elemento de resultado renderizado      | (Opcional)
        content: (data, source) => {
            source.innerHTML = data.match;
        },
        element: "li"
    },
    noResults: () => {                     // Script de acción en no hay resultados   | (Opcional)
        const result = document.createElement("li");
        result.setAttribute("class", "no_result");
        result.setAttribute("tabindex", "1");
        result.innerHTML = "No hay resultados";
        document.querySelector("#autoComplete_list").appendChild(result);
    },
    onSelection: feedback => {             // Script de acción en evento de selección | (Opcional)
        document.getElementById('autoComplete').value = feedback.selection.value;
    }
});
