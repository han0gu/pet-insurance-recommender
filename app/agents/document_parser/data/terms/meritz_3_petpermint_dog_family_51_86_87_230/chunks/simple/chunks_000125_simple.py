from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 규정에 따른 대출금과 보험료의 자동대출 납입 일의 다음날부터 그 다음 보험료의 납입최고(독촉)기간까지 의 '
 '이자(보험계약대출이율을 적용하여 계산)를 더한 금액이 해당 보험료가 납입된 것으로 계산한 해약환급금과 계약자 에게 지급할 기타 모든 '
 '지급금의 합계액에서 계약자의 회사 에 대한 모든 채무액을 뺀 금액을 초과하는 경우에는 보험 료의 자동대출납입을 더는 할 수 없습니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 76},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
