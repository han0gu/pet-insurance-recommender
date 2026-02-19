from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 청력장해는 순음청력검사 결과에 따라 데시벨(dB:decibel)로서 표시하고, 3회 이상 청력검사를 실시한 후 적용한다. 다만, 각 '
 '측정치의 결과값 차이가 ± 10dB 이상인 경우 청성뇌간반응검사(ABR)를 통해 객관적인 장해 상태를 재평가하여야 한다. 2) "한 귀의 '
 '청력을 완전히 잃었을 때" 라 함은 순음청력검사 결과 평균순음역치 가 90dB 이상인 경우를 말한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000880',
              'chunk_char_len': 209,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
