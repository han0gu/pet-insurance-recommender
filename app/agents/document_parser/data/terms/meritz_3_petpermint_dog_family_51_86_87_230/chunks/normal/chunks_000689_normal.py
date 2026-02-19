from langchain_core.documents import Document

chunk = Document(
    page_content=('지급기일의 61일 이후부터 90일 이내 기간 | 보험계약대출이율 + 가산이율(6.0%)\n'
 '지급기일의 91일 이후 기간 | 보험계약대출이율 + 가산이율(8.0%)\n'
 '만기환급금(보통약관 제10조 제1항) 및 해약환급금 (보통약관 제35조 제1항) (특별약관이 부가된 경우 특별약관의 해약환급금 포함) | '
 '지급사유가 발생한 날의 다음날부터 청구일까지의 기간 | 1년이내 : [보장]공시이율의 50%\n'
 '1년초과기간 : [보장]공시이율의 40%\n'
 '청구일의 다음날부터 지급일까지의 기간 | 보험계약대출이율'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 199},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000689',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
