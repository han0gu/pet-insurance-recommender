from langchain_core.documents import Document

chunk = Document(
    page_content=('【소멸시효】\n'
 '소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합 니다. 보험금 지급사유가 2023년 4월 1일에 발생하였음에 도 2026년 4월 '
 '1일까지 보험금을 청구하지 않는 경우 소 멸시효가 완성되어 보험금 등을 지급받지 못할 수 있습니 다.\n'
 '제42조(약관의 해석)\n'
 '\uf000 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하 여야 하며 계약자에 따라 다르게 해석하지 않습니다.\n'
 '【 신의성실의 원칙 】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 79},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
