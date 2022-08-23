Steve's No-Good-Very-Bad Jekyll Theme
=====================================

This is my custom Jekyll theme, which is basically [Joel Glovier](http://joelglovier.com/)'s `jekyll-new` theme smashed with [Alex King](http://www.alexking.org)'s [Favepersonal](https://crowdfavorite.com/favepersonal/) theme for Wordpress. I used Favepersonal for my Wordpress site before abandoning it. You can see my site at [svmiller.github.io](http://svmiller.github.io).

Much of what is contained in here is derivative of those two works. That said, do observe the `embedpdf.html` and `image.html` files in the `_includes` directory. `embedpdf.html` uses Google Docs to allow for embedding of PDF files hosted on Dropbox. `image.html` provides fancier images than what is standard for Markdown. An example use of `embedpdf.html` can be observed in the `cv.md` file. An example use of `image.html` can be observed in the `about.md` file.

I use data-driven navigation, which you can see in the `menu.yml` file in the `_data` directory. There's also a `nav.html` file in the `_includes` directory with modified `header.html`.

Mobile support is clearly functional, though some white-spacing could be improved. Feel free to offer improvements if you'd like.

`css` and `_sass` directories also functional, if a bit cluttered. Do observe new colors I created for `$clemson-orange` and `$clemson-purple` in `css/main.scss`.

Feel free to contact me at svmille@clemson.edu. Send along some cheers too if you find it useful.


# Deploying the website for yourself.

## Make a copy for your own repo. [Source](https://help.github.com/articles/duplicating-a-repository/)

1. Create the github repo you wish to use for your website. For our examples we will use `mrlucasch.github.io`
2. Clone the github template repo
```
git clone --bare git@github.com:mrlucasch/umass-website-template.git
```
3. Mirror-push to the new repo:
```
cd umass-website-template.git
git push --mirror git@github.com:mrlucasch/mrlucasch.github.io
```

## Customizing your website.

### Modify \_config.yml to control sitewide data
This file holds most of the generic site_wide settings for your jekyll site. It controls the domain name as well as some generic data.

1. Modify the title. This is the title that will appear at the top of your page.
```
title: Lucas Chaufournier
```
2. Add your email.
```
email: lucasch@cs.umass.edu
```
3. Edit the description. This appears at the bottom of the webpage on every page.
```
description: > # this means to ignore newlines until "baseurl:"
  "It is one of life's bitterest truths that bedtime so often arrives just when things are really getting interesting." -- Lemony Snicket
```
4. For `url` place the domain name for your website. For github pages this will be `mrlucasch.github.io`:
```
url: "http://mrlucasch.github.io"
```
5. Add your twitter_username if you feel so inclined:
```
twitter_username: lucasch
```
6. Add your github_username if you wish:
```
github_username:  mrlucasch
```
7. Feel free to modify your job title:
```
job.title: "Graduate Student"
```
That is it for the ```_config.yml``` file.

## Modify \_data/menu.yml to add Navigation links to the top.
This file contains the site layout for your webpage. It determines the links that appear in the main navigation bar.
Links are in the order that they will appear. Feel free to add more as you wish and remove others. Be mindful of spacing as yaml can be quite particular.

`href` is used to point to the markdown file for that link. The content for about would be located in about.md in the root directory. The content for item-1 is located in item-1.md in the folder resources.

If you want to have submenus, then you can follow the example of the resources page. To creates subfolders, just make a subfolder in the root directory with the parent link and add pages as necessary.


## Changing site content.
To change the content of each page you need to edit the corresponding markdown files. I break down the different pages below:

### index.html
This file contains the information you see on the main page when people navigate to your site. It is good to put a brief overview of yourself, your research, and selected publications. Also include an image of yourself so people know who you are!

To replace the default octocat, place your image in your /images/ folder and then modify the following line with the correct file name:
```
{% include image.html url="/images/NEW_FILE_NAME.PNG" caption="YOUR_NAME_HERE." width=300 align="center" %}

```
FYI: Despite being labeled with an .html extension, it should still accept regular markdown commands.

### about.md
This file contains more detailed information about you and your history. It is not as detailed as your full family tree but maybe include where you went to undergrad and some of your non-cs related hobbies. Feel free to find a happy medium between the start of time and just putting "I'm at umass". Don't be afraid to let your personality shine through!

You should also replace the octocat on this page. You can use the same boring picture that you do on your front page or you can spice things up with a picture of you skydiving or looking over some broad landscape. I don't know, you do you! This is just what I've seen more "seasoned" academics do.

### blog.md
This represents your blog and should not be altered much. Feel free to alter the Title at the top of the page to something more fun than "Blog". Also you can edit the tagline from:
```
Tell us about your blog. Hopefully it's cool.
```
to something more interesting. Leave the awful code there.

You may be wondering where the blog posts come from? Great Question! They are automatically pulled from the \_posts folder. We will go over that once we are done customizing the website.

### research.md
This is where you should really let your passion shine through. Go crazy and talk all about your research interests. Discuss what you are working on currently, where your passion for it comes from, and where you hope to push your research in the future. Really motivate the problems you are working on.

This is also a good place to put your full publications. If you are a superstar student and have boatloads of papers already then you may wish to break it off into it's own page or subpage. See the menu.html section to figure out how.

### cv.md
I am a little torn on ny feelings toward this page. Currently it just embeds a pdf from dropbox and allows you to download said pdf. I would rather it be an html version of your cv with a link to download the pdf version. I am working on a way to do this but styles in css are painful so for now we are stuck with this version. I will send an update when its up and working.

You may be wondering how to get the pdf part working. Simply upload your cv to dropbox and then create a shared link of the form `https://www.dropbox.com/s/ALPHANUMERICSTRING/fname.pdf` Then replace mine in the template. Easy as pie.

### resources.md and its subpages
This is for any misc information you wish to provide. I usually have 1 or 2 tutorial pages on how to use a technology that I need to reference all the time. You can see on my page I have reference guides for tmux, kvm, lxc, etc.

This is also a good place to keep pages about advice for others or links to good papers on how do things. This is totally up to you. If you don't feel like sharing with the world then simply remove it and edit the menu.yml page above.

## How do I add blog pages?
To write a blog post you simply need to create a new file in the \_posts folder.
## Activate Github pages for your repo

1. Navigate to your websites github repo
2. Click on settings.
3. Scroll down to `GitHub Pages`
4. Under source, select the `master branch` and then hit save.
5. Access your website from GITHUBUSERNAME.github.io
