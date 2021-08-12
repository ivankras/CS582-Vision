function loadFirstImageFromUrl() {
    const url = document.getElementById("firstimgUrl").value;
    addFirstImage(url);
}

function addFirstImage(src) {
    if (src === "") {
        alert("You should enter an image URL");
        return;
    }
    let img = document.createElement("img");
    img.src = src;
    const parentNode = document.getElementById("previewfirstimage");
    if (parentNode.children[0] != null) {
        parentNode.removeChild(parentNode.children[0]);
    }
    parentNode.appendChild(img);
}


function loadSecondImageFromUrl() {
    const url = document.getElementById("secondimgUrl").value;
    addSecondImage(url);
}

function addSecondImage(src) {
    if (src === "") {
        alert("You should enter an image URL");
        return;
    }
    let img = document.createElement("img");
    img.src = src;
    const parentNode = document.getElementById("previewsecondimage");
    if (parentNode.children[0] != null) {
        parentNode.removeChild(parentNode.children[0]);
    }
    parentNode.appendChild(img);
}


function display_result(src) {

    let img = document.createElement("img");
    img.src = src;
    const parentNode = document.getElementById("preview_result");
    if (parentNode.children[0] != null) {
        parentNode.removeChild(parentNode.children[0]);
    }
    parentNode.appendChild(img);
}

const axios = require('axios').default;

export const file = () => {

    const first_url = document.getElementById("firstimgUrl").value
    const second_url = document.getElementById("secondimgUrl").value
    const num_of_itr = document.getElementById("iterationRange").value

    axios.post('http://localhost:5000/api/getResult', {
        structure: first_url,
        style: second_url,
        iters: num_of_itr
    }).then(function (response) {
        console.log(response)
        display_result(response);
    }).catch(function (error) {
        console.log(error);
    });

};



