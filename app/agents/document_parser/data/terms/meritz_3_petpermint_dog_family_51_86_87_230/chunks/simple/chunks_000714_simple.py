from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) “심한 장해를 남긴 때”라 함은 순음청력검사 결과 평균순음역치가 80dB이상인 경우에 해당되어, 귀에다 대고 말하지 않고는 '
 '큰소리를 알아듣지 못하는 경우 를 말한다. 4) “약간의 장해를 남긴 때”라 함은 순음청력검사 결과'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 204},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000714',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
