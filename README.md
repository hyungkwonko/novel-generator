# Novel generation using BiLSTM with Attention model

## BiLSTM & Attention

There are many explanations on these method.

## Layer stacking

If you’re running Jekyll v3.3+ and **self-hosting** you can quickly install the theme as Ruby gem:

1. Add this line to your Jekyll site’s Gemfile:

    ```
    gem "jekyll-theme-cayman-blog"
    ```

2. Add this line to your Jekyll site’s _config.yml file:

    ```
    theme: jekyll-theme-cayman-blog
    ```

3. Then run Bundler to install the theme gem and dependencies:

    ```
    script/bootstrap
    ```

## 5 different models to choose next sentence

If you’re *hosting your blog with GitHub Pages* you’ll have to consider this:

:warning: As stated in the official [Jekyll documentation](https://jekyllrb.com/docs/themes/#installing-a-theme):

> If you’re publishing your Jekyll site on [GitHub Pages](https://pages.github.com/), note that GitHub Pages supports only some gem-based themes. See [Supported Themes](https://pages.github.com/themes/) in GitHub’s documentation to see which themes are supported.

Therefore, this theme, as well as many others, can not be installed in the same way as the ones officially supported by GitHub Pages (e.g. Cayman, Minima), a bit more effort has to be put on.

The easiest way I found to install _Cayman Blog Theme_, is [installing the theme gem](gem-install), and then [converting the gem-based theme to regular theme](https://jekyllrb.com/docs/themes/#converting-gem-based-themes-to-regular-themes).

Alternatively, for new projects, one could fork the whole theme, and keep only the interesting files.


## Further work - Genetic algorithm

This method is preferred for existing _Jekyll blogs_, as well as newly created ones. Notice that the files `index.md`, `about.md`, `contact.md` will be overwritten (only `index.md` is really needed, the other two are just placeholders).

1. Install the theme gem: ` $ gem install jekyll-theme-cayman-blog`
3. Run `$ gem env gemdir` to know where the gem was installed
4. Open the folder shown in the output
5. Open the folder `gems`
5. Open the theme folder (e.g. `jekyll-theme-cayman-blog-0.0.5`)
6. Copy all the files into your newly created or existing blog folder    
7. Leave empty `theme` your site's `_config.yml`:

    ```yml
    theme:
    ```
6. Modify `_config.yml`, `about.md`, `contact.md` for your project
7. [Customize the theme](customizing)
