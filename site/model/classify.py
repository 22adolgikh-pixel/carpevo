# classify.py — машинная разметка понятий-орнаментов по смысловым классам (фасет motif_class, статус auto).
# Классы — по отчёту о классификациях (Гулиева 2017, Тагиева 2015: геометрические / растительные / зооморфные /
# антропоморфные; остальное — рабочие обобщения) + служебный класс «структура ковра» для кайм, медальонов, полей.
# Правило: ключевые слова в русском/английском значении и в азербайджанском заголовке; классов может быть несколько.
import re

CLASSES = [
    # id, az, ru, en
    ('geometric', 'Həndəsi', 'Геометрические', 'Geometric'),
    ('vegetal', 'Nəbati', 'Растительные', 'Vegetal'),
    ('arabesque', 'İslimi–xətai', 'Арабески', 'Arabesque'),
    ('buta', 'Buta', 'Бута', 'Buta'),
    ('zoomorphic', 'Zoomorf', 'Животные', 'Zoomorphic'),
    ('bird', 'Quş', 'Птицы', 'Birds'),
    ('anthropomorphic', 'Antropomorf', 'Человек', 'Anthropomorphic'),
    ('mythical', 'Mifik', 'Мифические существа', 'Mythical creatures'),
    ('astral', 'Kosmoqonik', 'Небо и стихии', 'Cosmological'),
    ('objects', 'Məişət əşyaları', 'Предметы и постройки', 'Objects & buildings'),
    ('epigraphic', 'Epiqrafik', 'Надписи', 'Epigraphic'),
    ('tamga', 'Damğa', 'Тамги и знаки', 'Tamgas & signs'),
    ('structure', 'Xalçanın quruluşu', 'Части ковра', 'Carpet structure'),
]

KW = {
    'geometric': (r'ромб|треуг|квадрат|круг|окружн|восьмиуг|шестиуг|многоуг|угольн|крест|лини[яи]|полос|зигзаг|зуб|точк|сетк|решетк|решётк|спирал|завит|извит|вит[оа]|крюк|крюч|свастик|ступен|прямоуг|овал|дуг|петл|узел|плетен|косичк|цеп|звень|галун',
                  r'rhomb|diamond|triang|square|circle|octagon|hexagon|polygon|angular|cross|line|stripe|zigzag|tooth|serrat|dot|grid|lattice|spiral|twist|curl|hook|swastika|step|rectang|oval|arc|loop|knot|braid|plait|chain|weav|galloon',
                  r'bucaq|dairə|xətt|zolaq|dişli|burma|qıvrım|hörmə|qarmaq|çəngəl|zəncir'),
    'vegetal': (r'цвет|лист|ветв|ветк|дерев|роз[аы]|тюльпан|лили|ирис|мак|бутон|плод|ябло|гранат|алыч|айв|череш|виноград|лоз|куст|кипарис|гвозд|шишк|орех|миндал|каштан|трав|стеб|корен|семя|семен|тыкв|хлоп|сад|букет|колос|пшени|чинар|платан|тополь|ива|вишн|груш|инжир|хризант|фиалк|нарцисс|жасмин|лепест',
                r'flower|leaf|leaves|branch|tree|rose|tulip|lily|iris|poppy|bud|fruit|apple|pomegran|plum|quince|cherry|grape|vine|bush|cypress|nut|almond|chestnut|seed|pumpkin|cotton|blossom|floral|carnation|garden|bouquet|petal|plane tree|poplar|willow|pear|fig|chrysanth|violet|narcis|jasmin',
                r'gül|çiçək|yarpaq|budaq|ağac|alma|nar|ləçək|qönçə|şaxə|səbzə'),
    'arabesque': (r'ислими|хатаи|арабеск|бенди.?руми', r'arabesque|islimi|khatai', r'islimi|xətai|bəndi-?rumi'),
    'buta': (r'бут[аыуе]', r'\bbuta|boteh|paisley', r'buta'),
    'zoomorphic': (r'животн|овц|овеч|баран|ягн|коз[аыеёл]|козл|коров|бык|телён|лошад|кон[ья]|жереб|собак|щен|пёс|пес[^о]|кош[каеч]|верблюд|олен|джейран|газел|серн|лев|льв|тигр|барс|волк|лис[аиыц]|заяц|зайц|змея|змеи|змей|черепах|лягушк|рыб|краб|рак\b|скорпион|паук|насеком|бабочк|мотыл|жук|пчел|мух|лап[аык]|рог[аиу]?\b|рога|рожк|копыт|след|коготь|когт|хвост|шкур|кабан|медвед|ёж|еж',
                   r'animal|sheep|ram|lamb|goat|cow|bull|ox\b|horse|dog|puppy|hound|cat\b|camel|deer|gazelle|chamois|lion|tiger|leopard|wolf|fox|hare|snake|serpent|turtle|tortoise|frog|fish|crab|crayfish|scorpion|spider|insect|butterfly|moth|beetle|bee\b|fly\b|paw|horn|hoof|track|claw|tail|boar|bear|hedgehog',
                   r'heyvan|qoyun|keçi|inək|at\b|it\b|pişik|dəvə|maral|ceyran|aslan|şir\b|qurd|tülkü|dovşan|ilan|tısbağa|qurbağa|balıq|xərçəng|əqrəb|hörümçək|kəpənək|pərvanə|buynuz|dırnaq|pəncə'),
    'bird': (r'птиц|петух|петуш|кур[иа]|цыпл|павлин|соловей|соловь|голуб|орел|орёл|сокол|ястреб|коршун|утк|утин|гус[ьяи]|лебед|ласточк|ворон|сорок|фазан|турач|куропат|индейк|аист|журавл|сова|филин|попуга|крыл|пер[оья]\b|клюв|гнезд',
             r'bird|rooster|cock|hen\b|chicken|peacock|nightingale|dove|pigeon|eagle|falcon|hawk|kite \(bird|duck|goose|swan|swallow|crow|raven|magpie|pheasant|partridge|turkey|stork|crane|owl|parrot|wing|feather|beak|nest',
             r'quş|xoruz|toyuq|tovuz|tavus|bülbül|göyərçin|qartal|şahin|çalağan|ördək|qaz\b|qu quşu|qaranquş|qarğa|turac|kəklik|leylək|bayquş|qanad'),
    'anthropomorphic': (r'человек|люд[иея]|женщ|мужчин|мужск|женск|ребен|ребён|дет[иея]|девуш|девоч|невест|жених|мулл|старик|старух|сестр|брат|мать|матер|отец|отц|бабуш|дед|всадник|охотник|пастух|воин|голов[аы]\b|рук[аиу]|кист[ьи] рук|глаз|ног[аиу]|рот\b|губ[аыу]|бров|ребр|лиц[оа]|пальц|палец|локот|локт|пятипал|сердц|зуб[ыа]? |язык|ух[оа]\b|нос\b|бород|ус[ыа]\b|кос[аы]\b|волос|танц|хоровод|пляс',
                        r'human|man\b|men\b|woman|women|child|girl|boy|bride|groom|people|person|mullah|old man|sister|brother|mother|father|rider|horseman|hunter|shepherd|warrior|head\b|hand|eye|leg|foot|mouth|lip|brow|rib|face|finger|heart|tongue|ear\b|nose|beard|mustach|hair|dance',
                        r'adam|insan|qadın|kişi|uşaq|qız|gəlin|bəy|molla|baş\b|əl\b|göz\b|ayaq|ağız|dodaq|qaş\b|qabırğa|barmaq|ürək|dil\b|saç|hörük|yallı'),
    'mythical': (r'дракон|симург|феникс|пери\b|дэв|див\b|сказоч|мифич|грифон|ажда', r'dragon|simurgh|phoenix|griffin|mythic|fairy', r'əjdaha|əjdər|simurq|div\b|pəri'),
    'astral': (r'звезд|звёзд|солнц|лун[аыу]|месяц|неб[оа]|облак|туч|молни|гром|огон|огн|пламя|пламен|снег|снеж|дожд|радуг|ветер|ветр|вод[аы]\b|волн|озер|рек[аи]|мир[ау]? |вселен|космо|свет[аи]?\b|зар[яи]',
               r'star|sun\b|sunburst|moon|crescent|sky|cloud|lightning|thunder|fire|flame|snow|rain|rainbow|wind|water|wave|lake|river|cosmo|universe|light\b|dawn',
               r'ulduz|günəş|ay\b|göy\b|bulud|ildırım|şimşək|od\b|alov|qar\b|yağış|su\b|dalğa|çay\b'),
    'objects': (r'подсвеч|светильн|свеч|ламп|чайник|кувшин|чаш|тарел|блюд|поднос|ложк|нож\b|нож[аи]|ножниц|гребен|гребн|зеркал|сундук|колыбел|люльк|пояс|браслет|бус[ыи]|бисер|серьг|украшен|ожерел|ключ|замок|сито|решето|ковш|котел|котёл|сахарниц|маслен|очаг|мангал|кочерг|прялк|веретен|игл|наперст|шат[её]р|юрт|окн[оа]|двер|ворот|арк[аи]|купол|дом[аи]?\b|комнат|колес|телег|арб[аы]|лук\b|стрел|меч\b|сабл|секир|топор|кинжал|щит|копь|оружи|кальян|мангал|сосуд|горш|ваз|бутыл|коробоч|шкатул|сумк|мешоч|кисет|кисточк|кист[ьи]\b|бахром|подушк|ковер|ковёр|палас|одеял|скатерт|сурьм|флаг|знамя|корон|трон|мечеть|минарет|башн|мост|колодец|забор|стен',
                r'candle|lamp|teapot|jug|pitcher|ewer|bowl|plate|saucer|tray|spoon|knife|scissors|comb|mirror|chest|cradle|belt|bracelet|bead|earring|jewel|necklace|key|lock|sieve|ladle|cauldron|hearth|brazier|spinning|spindle|needle|thimble|tent|window|door|gate|arch\b|dome|house|room|wheel|cart|bow\b|arrow|sword|sabre|axe|dagger|shield|spear|weapon|hookah|vessel|pot\b|vase|bottle|box|bag|pouch|tassel|fringe|pillow|carpet|rug|blanket|tablecloth|kohl|flag|banner|crown|throne|mosque|minaret|tower|bridge|well\b|fence|wall',
                r'şamdan|çıraq|çaynik|küpə|kuzə|sini|nəlbəki|boşqab|qaşıq|bıçaq|qayçı|daraq|şana|güzgü|sandıq|beşik|kəmər|bilərzik|muncuq|sırğa|açar|qıfıl|xəlbir|ələk|qazan|ocaq|manqal|cəhrə|iynə|çadır|pəncərə|qapı|tağ\b|qübbə|ev\b|otaq|çarx|araba|kaman|ox\b|qılınc|balta|xəncər|qalxan|nizə|qəlyan|qotaz|bayraq|ələm|tac'),
    'epigraphic': (r'надпис|кетеб|картуш|письм|букв|каллиграф|стих', r'inscription|cartouche|letter|calligraph|verse|script', r'kitabə|kətəbə|yazı|hərf'),
    'tamga': (r'тамг|знак|печат|дамг|герб|клейм|оберег|амулет|талисман', r'tamga|sign\b|seal\b|emblem|coat of arms|stamp|amulet|talisman', r'damğa|möhür|gerb|tilsim|nişan'),
    'structure': (r'кайм|бордюр|медальон|гёль|гель\b|центральн(ое|ого) пол|срединн|угол\b|углов|рамк|обрамл|заставк|бахром', r'border|medallion|field\b|corner|frame|guard', r'haşiyə|mədaxil|bordür|göl\b|künc|xonça|orta sahə|ara sahə'),
}
# совпадение только с начала слова: иначе «ёж» находится в «снежный», «keçi» — в «keçirtmə»
B = r'(?:^|(?<=[^0-9a-zа-яёəıöüğşç]))'
RX = {k: [re.compile(B + '(?:' + p + ')', re.I) for p in v] for k, v in KW.items()}


def classify(c, kind_of_heads=()):
    """→ список классов; kind_of_heads — заголовки понятий, видом которых это понятие является (badam buta → buta)."""
    ru = ' '.join(filter(None, [c['meaning'].get('ru')] + [n['v'] for n in c['names'] if n['lang'] == 'ru' and n['kind'] not in ('meaning_rejected', 'misreading')]))
    en = ' '.join(filter(None, [c['meaning'].get('en')] + [n['v'] for n in c['names'] if n['lang'] == 'en' and n['kind'] != 'meaning_rejected']))
    az = ' '.join([c['headword']] + list(kind_of_heads))
    out = []
    for k, (rr, re_, ra) in RX.items():
        if rr.search(ru) or re_.search(en) or ra.search(az.lower()):
            out.append(k)
    if 'buta' in out:                       # бута — свой класс, «растительное» из-за слова «цветок» не добавляем
        out = [x for x in out if x not in ('vegetal',)] if not re.search(r'gül|çiçək', az.lower()) else out
    return out
