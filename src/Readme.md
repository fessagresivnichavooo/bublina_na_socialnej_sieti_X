## 📁 Štruktúra projektu

### 🔧 Zdrojový kód

- **`main.py`**  
  Hlavná časť programu. Koordinuje načítanie dát, spracovanie, analýzu a výstup výsledkov.

- **`AIAnalysis.py`**  
  Obsahuje nástroje na **obsahovú analýzu** tweetov a profilov.

- **`TwitterScrapper.py`**  
  Šablóna pre implementáciu rozhrania na **zber dát z Twitteru**.

- **`test_twitter_scrapper_from_json.py`**  
  Verzia scrappera, ktorá spracováva údaje zo súborov vo formáte ako `test_data_for_graph_*.json`. Umožňuje testovanie bez online pripojenia.

---

### 📂 Dátové súbory

- **`profile_analysis.json`**  
  Výsledky analýzy sledovaných profilov.

- **`tweet_analysis_test.json`**  
  Výstupy z analýzy tweetov – obsah, typ interakcie (odpoveď, retweet), sentiment atď.

- **`test_data_for_graph_*.json`**  
  Testovacie súbory so **surovými odpoveďami z backendu Twitteru (X)**.

- **`topic_translations.json`**  
  Cache pre **zero-shot klasifikáciu tém** – zabraňuje opakovaniu rovnakých výpočtov.

---

