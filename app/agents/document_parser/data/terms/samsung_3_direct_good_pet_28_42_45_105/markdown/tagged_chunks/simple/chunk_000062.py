from langchain_core.documents import Document

chunk = Document(
    page_content=('- 납입한 보험료를 계약자에게 돌려 드리며, 보험료 반환이 늦어진 기간에 대하여는 이\n'
 '- 계약의 보험계약대출이율을 연단위 복리로 계산한 금액을 더하여 지급합니다. 다만,\n'
 '- 계약자가 제1회 보험료를 신용카드로 납입한 계약의 청약을 철회하는 경우에는 회사\n'
 '- 는 청약의 철회를 접수한 날부터 3영업일 이내에 해당 신용카드회사로 하여금 대금\n'
 '- 청구를 하지 않도록 해야 하며, 이 경우 회사는 보험료를 반환한 것으로 봅니다.\n'
 '- ⑤ 청약을 철회할 때에 이미 보험금 지급사유가 발생하였으나 계약자가 그 보험금 지급'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
