from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 청력장해는 순음청력검사 결과에 따라 데시벨(dB : decibel)로서 표시하고 3회 이상 청력검사를 실시한 후 적용한다. 다만, '
 '각 측정치의 결과값 차이가 ±10dB 이상인 경우 청성뇌간반응검사(ABR)를 통해 객관적인 장해 상태를 재평가 하여야 한다. 2) “한 '
 '귀의 청력을 완전히 잃었을 때”라 함은 순음청력 검사 결과 평균순음역치가 90dB이상인 경우를 말한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 204},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000713',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
