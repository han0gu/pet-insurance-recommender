from langchain_core.documents import Document

chunk = Document(
    page_content=('| 14 | 구강, 치아, 혀, 악하선(턱밑샘), 이하선(귀밑샘) 및 설하선(혀밑샘) |\n'
 '| 15 | 귀[외이(바깥 귀), 고막, 중이(가운데귀), 내이(속귀), 청신경 및 유양돌기(꼭지돌기)포함] |\n'
 '| 16 | 안구 및 안구부속기[안검(눈꺼풀), 결막, 누기(눈물샘), 안근 및 안와내조직 포함] |\n'
 '| 17 | 신장 |\n'
 '| 18 | 부신 |\n'
 '| 19 | 요관, 방광 및 요도 |\n'
 '| 20 | 음경 |\n'
 '| 21 | 질 및 외음부 |\n'
 '| 22 | 전립선 |\n'
 '| 23 | 유방(유선 포함) |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'digestive', 'eye', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000738',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
