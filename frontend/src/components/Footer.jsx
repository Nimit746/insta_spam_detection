import React from 'react'
import { FaInstagram, FaTwitter, FaFacebook } from "react-icons/fa";

const Footer = () => {
    return (
        <footer className="w-[60%] mx-auto p-10 flex flex-col items-center justify-center gap-10">
            <div className="flex gap-60">
                <div className="flex">
                    <h3 className='text-lg font-bold'>Privacy Policy</h3>
                </div>
                <div className="flex flex-col gap-5">
                    <div className="flex ">
                        <h3 className='text-lg font-bold'>Terms of Service</h3>
                    </div>
                    <div className="flex gap-x-5 items-center justify-center">
                        <FaTwitter size={20}/>
                        <FaFacebook size={20}/>
                        <FaInstagram size={20}/>
                    </div>
                </div>
                <div className="flex">
                    <h3 className='text-lg font-bold'>Contact Us</h3>
                </div>
            </div>
            <div className="flex">
                <h3 className='font-bold'> &copy; {new Date(Date.now()).getFullYear()} SpamGuard. All rights Reserved.</h3>
            </div>
        </footer>
    );
}

export default Footer