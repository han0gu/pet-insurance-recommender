from langchain_core.documents import Document

chunk = Document(
    page_content=('잔여기간을 보험기간으로 합니다.\n'
 '\uf000 회사는 갱신계약에 대하여 갱신전 약관을 적용하며, 보\n'
 '험요율에 관한 제도 또는 보험료 등을 개정한 경우에는 갱\n'
 '신계약에 대해서는 갱신일 현재의 제도 또는 보험료 등을\n'
 '적용합니다.\n'
 '\uf000 회사는 제1항의 갱신제한 사유 및 제3항의 갱신계약 보\n'
 '험료에 대하여 갱신전 계약의 보험기간이 끝나기 15일 전까\n'
 '지 그 내용을 계약자에게 서면, 전화 또는 전자문서 등으로\n'
 '안내합니다.\n'
 '\uf000 제3항 및 제4항에도 불구하고 법령 및 표준약관 변경으'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000516',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
