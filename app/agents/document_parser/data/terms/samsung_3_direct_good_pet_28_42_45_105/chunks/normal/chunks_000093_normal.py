from langchain_core.documents import Document

chunk = Document(
    page_content=('⑥ 계약자가 보험수익자를 변경하지 않고 사망한 때에는 계약자 사망시점에 지정되어 있 는 보험수익자의 권리가 확정됩니다. 그러나 계약자가 '
 '사망한 이후 그 승계인이 보험 수익자를 변경할 수 있다는 별도의 약정이 있는 경우에는 승계받은 계약자가 보험수 익자를 변경할 수 '
 '있습니다. ⑦ 회사는 제1항 제4호에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약 관을 드리고, 변경된 계약자가 '
 '요청하는 경우 약관의 중요한 내용을 설명하여 드립니 다.\n'
 '제22조 (보험나이 등)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
