import React from 'react'
import Button from './Button';
import { Link } from 'react-router-dom';


const Navbar = () => {



    return (
        <header className='border-b-2 border-b-[#835061] flex items-center justify-between gap-10 w-full py-2 px-15'>
            <nav className="flex items-center justify-between gap-10 w-screen">
                <div className="flex items-center justify-center gap-2">
                    <img />
                    <h1 className='text-2xl font-bold'>SpamGuard</h1>
                </div>

                <div className='flex items-center justify-center'>
                    <ul className='flex items-center justify-center gap-10'>
                        <li><a href='/'>Home</a></li>
                        <li><a href='/'>Features</a></li>
                        <li><a href='/'>Pricing</a></li>
                        <li><a href='/'>Support</a></li>
                    </ul>
                    <Button title='Get Started' />
                </div>
            </nav>
        </header>
    );
}

export default Navbar
