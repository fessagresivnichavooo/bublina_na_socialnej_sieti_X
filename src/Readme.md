## 📁 Štruktúra projektu

### 🔧 Zdrojový kód

- **`main.py`**  
  Hlavná časť programu. Koordinuje načítanie dát, spracovanie, analýzu a výstup výsledkov.

- **`AIAnalysis.py`**  
  Obsahuje nástroje na **obsahovú analýzu** tweetov a profilov (klasifikácia tém, analýza sentimentu, práca so sieťami).

- **`TwitterScrapper.py`**  
  Šablóna pre rozhranie na **zber dát z Twitteru**, buď pomocou API alebo web scrapingu.

- **`test_twitter_scrapper_from_json.py`**  
  Verzia scrappera, ktorá spracováva údaje zo súborov (napr. `test_data_for_graph_*.json`). Umožňuje testovanie bez online pripojenia.

---

### 📂 Dátové súbory

- **`profile_analysis.json`**  
  Výsledky analýzy profilov – napr. bio, lokalita, sledovatelia/sledovaní.

- **`tweet_analysis_test.json`**  
  Výstupy z analýzy tweetov – obsah, typ interakcie (odpoveď, retweet), sentiment, témy, zmienky atď.

- **`test_data_for_graph_*.json`**  
  Testovacie súbory so **surovými odpoveďami z backendu Twitteru (X)**. Slúžia na simuláciu API vstupu.

- **`topic_translations.json`**  
  Cache pre **zero-shot klasifikáciu tém** – zabraňuje opakovaniu rovnakých výpočtov.

---

