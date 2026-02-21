from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 계약자, 피보험자가 동일하고 보험수익자가 계약자의\n'
 '- 법정상속인인 계약일 경우\n'
 '\uf000 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입\n'
 '한 보험료를 계약자에게 돌려 드리며, 보험료를 받은 기간\n'
 '에 대하여 보험계약대출이율을 연단위 복리로 계산한 금액\n'
 '을 더하여 지급합니다.- \n'
 '# 【보험계약대출이율】계약자는 해당 계약의 해약환급금 범위내에서 회사가 정\n'
 '한 방법에 따라 대출을 받을 수 있는데, 이를「보험계약\n'
 '대출」이라 합니다. 이 때 적용되는 이율을「보험계약대66출이율」이라 하며, 회사에서 별도로 정한 방법에 따라'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000078',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
