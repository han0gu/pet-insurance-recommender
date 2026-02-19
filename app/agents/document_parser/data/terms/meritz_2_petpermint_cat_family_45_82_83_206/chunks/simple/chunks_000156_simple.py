from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 이 계약에 대하여 계약자에게 배당금을 지급하지 않 습니다.\n'
 '제38조(중도인출)\n'
 '\uf000 계약자는 보장개시일부터 2년 이상 지난 유효한 계약으 로서 계약자의 요청이 있는 경우에 한하여 보험연도 기준 연4회에 한하여 '
 '중도인출 할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 78},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 134,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
