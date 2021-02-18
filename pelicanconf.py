AUTHOR = 'Gastón Amengual'
SITENAME = 'Gastón Amengual'
SITEURL = ''
PATH = 'content'
AUTHOR_INFO = ''
TIMEZONE = 'America/Argentina/Buenos_Aires'
AUTHOR_WEB = ''
AUTHOR_AVATAR = ''

DEFAULT_LANG = 'en'

RELATIVE_URLS = True

THEME = 'MINIMALXY'
START_YEAR = 2021
CURRENT_YEAR = 2021

CATEGORIES_SAVE_AS = 'categories.html'
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_PAGES_ON_MENU = False
MENUITEMS = (
    ('Home', "/index.html"), 
    ('About', '/pages/about.html'),
    ('Portfolio', '/pages/portfolio.html'),
    ('Articles', "/" + CATEGORIES_SAVE_AS),
)

GOOGLE_ANALYTICS = "G-6KC62F9717"