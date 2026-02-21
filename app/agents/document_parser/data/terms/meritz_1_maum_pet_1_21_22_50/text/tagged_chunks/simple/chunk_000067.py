from langchain_core.documents import Document

chunk = Document(
    page_content=('지체없이 알려야 합니다.\n'
 '④ 계약자가 청약을 철회한 때에는 회사는 청약의 철회를 접수한 날부터 3영업일 이내에\n'
 '납입한 보험료를 계약자에게 돌려드리며, 보험료 반환이 늦어진 기간에 대하여는 ‘보험\n'
 '개발원이 공시하는 보험계약대출이율’을 연단위 복리로 계산한 금액을 더하여 지급합니- 12 -다. 다만, 계약자가 제1회 보험료 등을 '
 '신용카드로 납입한 계약의 청약을 철회하는 경\n'
 '우에는 회사는 청약의 철회를 접수한 날부터 3영업일 이내에 해당 신용카드회사로 하'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
