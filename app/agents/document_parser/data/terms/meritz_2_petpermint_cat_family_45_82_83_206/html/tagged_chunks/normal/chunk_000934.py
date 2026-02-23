from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 각 측정치의 결과값 차이가<br>±10dB 이상인 경우 청성뇌간반응검사(ABR)를 통해<br>객관적인 장해 상태를 재평가 '
 '하여야 한다.<br>2) “한 귀의 청력을 완전히 잃었을 때”라 함은 순음청력<br>검사 결과 평균순음역치가 90dB이상인 경우를 '
 '말한다.<br>3) “심한 장해를 남긴 때”라 함은 순음청력검사 결과<br>평균순음역치가 80dB이상인 경우에 해당되어, '
 '귀에다<br>대고 말하지 않고는 큰소리를 알아듣지 못하는 경우<br>를 말한다.<br>4) “약간의 장해를 남긴 때”라 함은 순음청력검사'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000934',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
