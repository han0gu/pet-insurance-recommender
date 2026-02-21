from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 보장관련 보험금 (보통약관 제3조) (특별약관이 부가된 경우 특별약관의 보험금 포함) | 지급기일의 다음 날부터 30일 이내 기간 | '
 '보험계약대출이율 |\n'
 '| 보장관련 보험금 (보통약관 제3조) (특별약관이 부가된 경우 특별약관의 보험금 포함) | 지급기일의 31일 이후부터 60일 이내 기간 '
 '| 보험계약대출이율 + 가산이율(4.0%) |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
