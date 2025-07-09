let email=document.querySelector('.email');
let password=document.querySelector('.password');

const login=document.querySelector('.login-bt');
login.addEventListener('click',()=>{
    alert("Login Successfull!!");
    email.value="";
    password.value="";
});