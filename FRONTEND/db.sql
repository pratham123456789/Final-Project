drop database if exists plant;
create database plant;
use plant;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(225),
    email VARCHAR(50),
    password VARCHAR(50)
    );
