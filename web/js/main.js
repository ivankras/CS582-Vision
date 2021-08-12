
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


