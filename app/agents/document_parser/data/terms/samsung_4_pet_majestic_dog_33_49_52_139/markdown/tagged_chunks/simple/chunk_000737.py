from langchain_core.documents import Document

chunk = Document(
    page_content=('| 4 | 직장 |\n'
 '| 5 | 항문 |\n'
 '| 6 | 간 |\n'
 '| 7 | 담낭(쓸개) 및 담관 |\n'
 '| 8 | 췌장 |\n'
 '| 9 | 비장 |\n'
 '| 10 | 기관, 기관지, 폐, 흉막 및 흉곽(늑골 포함) |\n'
 '| 11 | 코[외비(코 바깥), 비강(코 안) 및 부비강(코 곁굴) 포함] |\n'
 '| 12 | 인두 및 후두(편도 포함) |\n'
 '| 13 | 식도 |\n'
 '| 14 | 구강, 치아, 혀, 악하선(턱밑샘), 이하선(귀밑샘) 및 설하선(혀밑샘) |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000737',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
