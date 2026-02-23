from langchain_core.documents import Document

chunk = Document(
    page_content=('나를 충족하는 때에는 자필서명을 생략할 수 있으며, 제2항의 규정에 따른 음성녹음\n'
 '내용을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를 전달한 것\n'
 '으로 봅니다.- 1. 계약자, 피보험자 및 보험수익자가 동일한 계약의 경우\n'
 '- 2. 계약자, 피보험자가 동일하고 보험수익자가 계약자의 법정상속인인 계약일 경우\n'
 '# ⑤ 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게 돌려드리며, 보험료를 받은 기간에 대하여 이 계약의 '
 '보험계약대출이율을 연단위 복리로'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000088',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
