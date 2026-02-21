from langchain_core.documents import Document

chunk = Document(
    page_content=('버 제2조 제2호에 따르 전자서명이 이느 경으로 서 사버 시해려 제44조의2에 정차는- \n'
 '- 36 -- ⑥ 계약자가 보험수익자를 변경하지 않고 사망한 때에는 계약자 사망시점에 지정되어 있\n'
 '- 는 보험수익자의 권리가 확정됩니다. 그러나 계약자가 사망한 이후 그 승계인이 보험\n'
 '- 수익자를 변경할 수 있다는 별도의 약정이 있는 경우에는 승계받은 계약자가 보험수\n'
 '- 익자를 변경할 수 있습니다.\n'
 '- ⑦ 회사는 제1항 제4호에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000078',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
