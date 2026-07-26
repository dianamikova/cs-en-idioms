# Experiment 1 — Robustness Test Results

These four tests check the KB-CoT translation pipeline beyond the main
ablation: whether it behaves deterministically, whether it correctly does
nothing on non-idiomatic input, and whether it can be misled by incorrect
injected knowledge on sentences that share vocabulary with real idioms but
are genuinely literal. Test A confirms reproducibility on the idiom
dataset itself. Test B confirms the pipeline stays inert on ordinary
sentences. Tests C and D probe false-positive risk under increasingly
adversarial conditions, using deliberately wrong injected meanings. All
runs use the frozen, 4-bit quantized EuroLLM-9B-Instruct with deterministic
(greedy) decoding, seed=42.

---

## Test A — Determinism (idiomatic dataset, 20 examples)

Command: `!python scripts/translate.py --limit 20 --output results/exp1_20_examples.jsonl`

| # | id | CS | Direct | KB-CoT |
|---|---|---|---|---|
| 1 | 100 | Koledoval si o pěkný průšvih, když omylem odešel z obchodu bez zaplacení. | He was asking for a big trouble when he left the store without paying. | He was asking for trouble when he accidentally left the store without paying. |
| 2 | 18 | Nejdříve jen brnkal sociální demokracii na nervy. | At first, he just tickled the nerves of the social democrats. | At first, he was just annoying the social democrats. |
| 3 | 5 | Stojí za svým názorem stůj co stůj. | He stands by his opinion no matter what. | He stands by his opinion at all costs. |
| 4 | 41 | Šéf přivřel oko nad tím, že přišel kolega o pět minut pozdeji. | The boss closed an eye over the fact that the colleague came five minutes later. | The boss turned a blind eye to the fact that the colleague came five minutes later. |
| 5 | 37 | Šéf mu dal padáka kvůli častým pozdním příchodům. | The boss fired him because of his frequent late arrivals. | The boss fired him because of his frequent late arrivals. |
| 6 | 33 | Slibný projekt se ukázal být danajským darem, protože nás stál všechny zaměstnance. | The promising project turned out to be a Trojan horse, because it cost us all employees. | The promising project turned out to be a Trojan horse because it cost all of us employees. |
| 7 | 22 | Úřady jsou na prokazování pochybného otcovství krátké. | The authorities are short of proving doubtful paternity. | The authorities are short of proving doubtful paternity. |
| 8 | 16 | Zvládni maturitu z českého jazyka levou zadní! | Pass your Czech language matriculation exam with flying colors! | Pass your Czech language matriculation exam with ease! |
| 9 | 106 | Není možné vynášet soudy s nedostatečnými informacemi. | It is not possible to make judgments with insufficient information. | It is not possible to make judgments with insufficient information. |
| 10 | 85 | Uspořádali jsme naší kamarádce rozlučku se svobodou. | We organized a bachelorette party for our friend. | We organized a bachelorette party for our friend before her marriage. |
| 11 | 14 | Vysvětlovat mu, že nemá věřit dezinformacím, je jako bojovat s větrnými mlýny. | Explaining to him that he should not believe in disinformation is like fighting with windmills. | Explaining to him that he should not believe in disinformation is like fighting with windmills. |
| 12 | 92 | Mám plné zuby toho tého vymlouvání se na počasí. | I am fed up with this excuse about the weather. | I am sick and tired of all this excusing oneself on the weather. |
| 13 | 64 | To si nemůžeš pamatovat, to jsi ještě tahal kačera. | You can't remember that, you were still pulling the duck. | You can't remember that, you were still pulling the duck. |
| 14 | 6 | Chce uspět za každou cenu. | He wants to succeed at any cost. | He wants to succeed at all costs. |
| 15 | 5 | Stůj co stůj se snažil nevypadat hloupě. | No matter what, he tried not to look stupid. | At all costs he tried not to look stupid. |
| 16 | 15 | Bereš to moc na lehkou váhu. | You take it too lightly. | You are underestimating it. |
| 17 | 32 | Když jsme byli mladí, občas jsme chodili za školu, protože jsme si chtěli užít sluníčko. | When we were young, we sometimes skipped school because we wanted to enjoy the sun. | When we were young, we sometimes skipped school because we wanted to enjoy the sun. |
| 18 | 35 | Chodila jsem s ním skoro šest let, ale nakonec jsem mu dala košem. | I went out with him for almost six years, but in the end I turned him down. | I went out with him for almost six years, but in the end I turned him down. |
| 19 | 78 | Musíš si být víc jistý sám sebou, aby to z tebe nelezlo jak z chlupaté deky. | You have to be more confident in yourself so that it doesn't come out of you like from a furry blanket. | You have to be more confident in yourself so that it doesn't come out of you like from a hairy blanket. |
| 20 | 94 | Plný brýle toho měl už po deseti minutách. | He was already full of glasses after ten minutes. | He was sick and tired of it after ten minutes. |

### Evaluation

*(to be added)*

---

## Test B — Determinism and Isolation (non-idiomatic dataset, 10 examples)

Command: `!python scripts/translate.py --input data/general_test.jsonl --output results/general_results.jsonl`

Dataset: `data/general_test.jsonl`

| # | id | CS | Direct | KB-CoT |
|---|---|---|---|---|
| 1 | gen1 | Šel jsem do lesa a potkal jsem tam vlka. | I went to the forest and met a wolf there. | I went to the forest and met a wolf there. |
| 2 | gen2 | Ráno jsem si uvařil čaj a dal jsem si snídani. | I made tea and had breakfast in the morning. | I made tea and had breakfast in the morning. |
| 3 | gen3 | Ty boty mi byly malé, tak jsem je musela jít vrátit. | Those shoes were too small for me, so I had to return them. | Those shoes were too small for me, so I had to return them. |
| 4 | gen4 | Ráda si ráno udělám kávu z výběrových zrn. | I like to make coffee from selected grains in the morning. | I like to make coffee from selected grains in the morning. |
| 5 | gen5 | V Praze bylo horko, tak jsme jeli do hor. | It was hot in Prague, so we went to the mountains. | It was hot in Prague, so we went to the mountains. |
| 6 | gen6 | V parku jsme s kamarádkou cvičily jógu, dokud nezačalo pršet. | We were doing yoga in the park with a friend until it started to rain. | We were doing yoga in the park with a friend until it started to rain. |
| 7 | gen7 | Chtěl bych na dovolenou do Itálie, ale nemám peníze. | I would like to go on holiday to Italy, but I don't have any money. | I would like to go on holiday to Italy, but I don't have any money. |
| 8 | gen8 | Chtěli bychom jít do kina, ale lístky jsou beznadějně vyprodány. | We would like to go to the cinema, but tickets are hopelessly sold out. | We would like to go to the cinema, but tickets are hopelessly sold out. |
| 9 | gen9 | Až budeme doma, budeme stavět gauč. | When we are at home, we will build a couch. | When we are at home, we will build a couch. |
| 10 | gen10 | Museli jsme přikrýt ten keř síťkou, aby na něj nelítali ptáci. | We had to cover the bush with a net so that birds wouldn't fly to it. | We had to cover the bush with a net so that birds wouldn't fly to it. |

### Evaluation

*(to be added)*

---

## Test C — False-Positive with KB-CoT (14 examples, easy set)

Testing whether KB-CoT causes a wrong/idiomatic translation on genuinely
literal sentences that share vocabulary with real idioms.

| # | id | Status | CS | Injected meaning | Direct | KB-CoT |
|---|---|---|---|---|---|---|
| 1 | gen1 | PASS | Máme dveře s kulatou klikou, ale vůbec se mi nelíbí. | to have good luck; be fortunate | We have doors with a round handle, but I don't like it at all. | We have a door with a round handle, but I don't like it at all. |
| 2 | gen2 | PASS | Když u nás byla na návštěvě, poručila si vodu místo vína, které jsme dovezli z Francie. | to demand of others standards you ignore yourself; hypocrisy | When she was visiting us, she ordered water instead of wine that we brought from France. | When she visited us, she ordered water instead of wine that we brought from France. |
| 3 | gen3 | PASS | Na zalití skleníku stačila voda z konve. | to rain very heavily | It was enough water from the watering can to water the greenhouse. | The water from the watering can was enough to water the greenhouse. |
| 4 | gen4 | PASS | Když mraky odkryly slunce, všichni rázem přimhouřili oči, takže ta fotka moc hezky nevypadá. | to intentionally ignore or forgive a minor fault; turn a blind eye | When the clouds uncovered the sun, everyone immediately squinted their eyes, so the photo doesn't look very nice. | When the clouds uncovered the sun, everyone immediately squinted their eyes, so the photo doesn't look very nice. |
| 5 | gen5 | PASS | Babičce jsme kdysi koupili lístky na Zagorku, a byla z toho nadšená. | to overreact dramatically, soap-opera style | We once bought tickets to Zagorka for our grandmother and she was thrilled. | We once bought tickets to Zagorka for our grandmother and she was thrilled. |
| 6 | gen6 | PASS | Včera jsem si dal kopačky před dveře a ráno byly pryč. | to break up with or reject a romantic partner | Yesterday I put my shoes in front of the door and in the morning they were gone. | Yesterday I put my foot in the door and in the morning they were gone. |
| 7 | gen7 | PASS | V letadle mi dali na záda padák a šli jsme skákat. | to be fired or dismissed from one's job | They gave me a parachute on my back in the plane and we went to jump. | They gave me a parachute on the plane and we went to jump. |
| 8 | gen8 | PASS | Dej mi ten koš, ať ho vynesu, než přijedou popeláři. | to reject or turn down a romantic proposal | Give me the trash can so I can take it out before the garbage collectors arrive. | Give me the trash can so I can take it out before the garbage collectors arrive. |
| 9 | gen9 | PASS | Neházej všechny ty brambory do pytle, ať nezplesniví. | to treat distinct things or people as if they were all the same; to lump together | Don't put all those potatoes in the bag, or they will go bad. | Don't put all those potatoes in a bag, or they will rot. |
| 10 | gen10 | PASS | Ty brýle vůbec nefungují, protože je mám věčně plné vody. | to be fed up with something; sick and tired of it | The glasses don't work at all, because they are always full of water. | Those glasses don't work at all, because I have them full of water all the time. |
| 11 | gen11 | CONTROL | Sestra připevnila sondu k srdci, aby udělala snímek. | — | The nurse attached a probe to the heart to take a picture. | The nurse attached a probe to the heart to take a picture. |
| 12 | gen12 | CONTROL | Poslali jsme ho k řece s vědrem pro vodu. | — | We sent him to the river with a bucket for water. | We sent him to the river with a bucket for water. |
| 13 | gen13 | PASS | Má jedno ucho větší než to druhé. | to be listening very attentively; all ears | One ear is bigger than the other. | He has one ear bigger than the other. |
| 14 | gen14 | PASS | Když jsem byl malý kluk, spadl jsem z trakaře. | things falling into chaos or disorder | When I was a little boy, I fell from a wheelbarrow. | When I was a little boy, I fell off a wheelbarrow. |

### Evaluation

*(to be added)*

---

## Test D — False-Positive with KB-CoT (10 examples, hard set)

Sentences deliberately made harder to resist — several use the idiom's
own vocabulary or exact surface phrasing in a genuinely literal context.

Command: `!python scripts/test_kbcot_false_pos.py --input data/false_pos_test_2.jsonl --output results/false_pos_test_2_results.jsonl`

| # | id | Status | CS | Injected meaning | Direct | KB-CoT | Found |
|---|---|---|---|---|---|---|---|
| 1 | gen1 | FLAGGED | Máš tu kliku ve skladu nebo ne?. | to have good luck; be fortunate | Do you have a key in the warehouse or not? | Do you have a lucky break in the warehouse or not? | lucky, luck |
| 2 | gen2 | PASS | Většinou piju víno místo vody, když jsem na kázání. | to demand of others standards you ignore yourself; hypocrisy | I usually drink wine instead of water when I am at a sermon. | Mostly I drink wine instead of water when I am at a sermon. | — |
| 3 | gen3 | PASS | Z konve lilo hodně vody. | to rain very heavily | A lot of water was pouring from the watering can. | It was raining very heavily. | — |
| 4 | gen4 | FLAGGED | Nesmíš přimhouřit oko, jinak se ti tam ta kapka nedostane. | to intentionally ignore or forgive a minor fault; turn a blind eye | You mustn't close an eye, otherwise the drop won't get there. | You must not turn a blind eye, otherwise that drop will not get there. | blind eye, turn a blind |
| 5 | gen5 | PASS | Kdysi dělal pro Zagorku, něco jako marketing. | to overreact dramatically, soap-opera style | He used to work for Zagorka, something like marketing. | He used to do marketing for Zagorka. | — |
| 6 | gen6 | PASS | Při resuscitaci polož dlaň na úroveň srdce. | to weigh on someone's mind; matter deeply to someone | During resuscitation, place your palm at the level of the heart. | When resuscitating, place your palm at the level of the heart. | — |
| 7 | gen7 | PASS | Dělali hada z plastelíny. | to form a conga line; move in a snaking line together | They made a snake from plasticine. | They made a snake from plasticine. | — |
| 8 | gen8 | PASS | Jakmile vystřelil po bažantovi, hodil flintu do žita a běžel ho hledat do pole. | to give up; abandon an effort prematurely | As soon as he shot at a pheasant, he threw the rifle into the wheat and ran to look for him in the field. | As soon as he shot at a pheasant, he threw the rifle into the wheat and ran to look for him in the field. | — |
| 9 | gen9 | PASS | Neházej tu zeleninu do jednoho pytle, ať to můžeme zvážit u pokladny. | to treat distinct things or people as if they were all the same; to lump together | Don't put all the vegetables in one bag so we can weigh them at the cash register. | Don't put all the vegetables in one bag so we can weigh them at the checkout. | — |
| 10 | gen10 | PASS | Tu kulku má blízko srdci, málem ho trefila. | to hold something dear; care deeply about | He has the bullet close to his heart, it almost hit him. | He holds that bullet dear; it almost hit him. | — |

### Evaluation

*(to be added)*

---

## Evaluation

*(to be added)*
