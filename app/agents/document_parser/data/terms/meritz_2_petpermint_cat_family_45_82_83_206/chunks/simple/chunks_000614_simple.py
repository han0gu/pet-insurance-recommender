from langchain_core.documents import Document

chunk = Document(
    page_content=('청구일의 다음날부터 지급일까지의 기간 | 보험계약대출이율\n'
 '주) 1. 회사가 만기환급금의 지급시기 도래 7일 이전에 지급 사유와 금액을 알리지 않은 경우, 지급사 유가 발생한 날의 다음 날부터 '
 '청구일까지의 기 간은 [보장]공시이율을 적용하여 계산한 이자를 지급합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 174},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000614',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
