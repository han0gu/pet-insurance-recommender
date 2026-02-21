from langchain_core.documents import Document

chunk = Document(
    page_content=('- 법」제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에 정하는\n'
 '- 바에 따라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함)으로\n'
 '- 동의하여야 합니다.\n'
 '- ⑥ 계약자가 보험수익자를 변경하지 않고 사망한 때에는 계약자 사망시점에 지정되어 있\n'
 '- 는 보험수익자의 권리가 확정됩니다. 그러나 계약자가 사망한 이후 그 승계인이 보험\n'
 '- 수익자를 변경할 수 있다는 별도의 약정이 있는 경우에는 승계받은 계약자가 보험수\n'
 '- 익자를 변경할 수 있습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000097',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
