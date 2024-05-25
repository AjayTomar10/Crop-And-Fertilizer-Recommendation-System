document.addEventListener('DOMContentLoaded', (event) => {
    var modal = document.getElementById('loginModal');
    var signinBtn = document.getElementById('signin');
    var span = document.getElementsByClassName('close')[0];

    signinBtn.onclick = function() {
        modal.classList.add("block");
        modal.classList.remove("hidden");
        // Show the login form by default when the modal is opened
        document.getElementById('loginForm').style.display = "block";
        document.getElementById('registerForm').style.display = "none";
        document.querySelector('.tab button:first-child').classList.add("active");
    }

    span.onclick = function() {
        modal.classList.remove("block");
        modal.classList.add("hidden");
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.classList.remove("block");
            modal.classList.add("hidden");
        }
    }

    // Default to show the login form
    document.querySelector('.tab button:first-child').click();
});

function openForm(evt, formName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(formName).style.display = "block";
    evt.currentTarget.className += " active";
}
