from langchain_core.documents import Document

chunk = Document(
    page_content=('도의 계약해당일이 없는 경우에는 해당 월의 마지막 날을\n'
 '계약해당일로 합니다.예시1) 계약일 : 2020년 10월 1일\n'
 '-> 계약해당일 : 10월 1일\n'
 '예시2) 계약일 : 2020년 2월 29일\n'
 '-> 계약해당일 : 2월 말일# 제25조(계약의 소멸)\uf000 제3조(보험금의 지급사유)에서 정한 일반상해80%이상후\n'
 '유장해보험금 지급사유가 발생한 경우에는 이 보장책임은\n'
 '그 때부터 소멸됩니다.\n'
 '\uf000 제1항에 따라 이 계약의 보장책임이 소멸된 때에는 회사\n'
 '는 이 보장책임의 해약환급금을 지급하지 않으며, 그 때까'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
