from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) "심한 장해를 남긴 때" 라 함은 순음청력검사 결과 평균순음역치가 80dB 이상 인 경우에 해당되어, 귀에다 대고 말하지 않고는 '
 '큰 소리를 알아듣지 못하는 경우를 말한다. 4) "약간의 장해를 남긴 때" 라 함은 순음청력검사 결과 평균순음역치가 70dB 이 상인 '
 '경우에 해당되어, 50cm 이상의 거리에서는 보통의 말소리를 알아듣지 못하는 경우를 말한다'),
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
 'indexing': {'chunk_id': 'chunk_000881',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
