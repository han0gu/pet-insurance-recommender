from langchain_core.documents import Document

chunk = Document(
    page_content=('이상 청력검사를 실시한 후 적용한다. 다만, 각 측정치의 결과값 차이가\n'
 '± 10dB 이상인 경우 청성뇌간반응검사(ABR)를 통해 객관적인 장해 상태를\n'
 '재평가하여야 한다.\n'
 '2) "한 귀의 청력을 완전히 잃었을 때" 라 함은 순음청력검사 결과 평균순음역치\n'
 '가 90dB 이상인 경우를 말한다.\n'
 '3) "심한 장해를 남긴 때" 라 함은 순음청력검사 결과 평균순음역치가 80dB 이상\n'
 '인 경우에 해당되어, 귀에다 대고 말하지 않고는 큰 소리를 알아듣지 못하는\n'
 '경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000744',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
