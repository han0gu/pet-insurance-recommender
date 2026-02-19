from langchain_core.documents import Document

chunk = Document(
    page_content=('【보험안내자료】\n'
 '계약의 청약을 권유하기 위해 만든 자료 등을 말합니다.\n'
 '제44조(법령 등의 개정에 따른 계약내용의 변경)\n'
 '\uf000 회사는 보험금 지급사유 관련 법률이 개정된 경우에는 변경된 내용을 적용합니다. \uf000 제1항에도 불구하고 다음 각 호 '
 '중 어느 한 가지에 해당 되는 경우에는 회사는 객관적이고 합리적인 범위내에서 기 존 계약내용에 상응하는 새로운 보장내용으로 계약내용을 '
 '변경할 수 있습니다.\n'
 '① 관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한 보험금 지급사유 판정기준이 변경되는 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 84},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000167',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
