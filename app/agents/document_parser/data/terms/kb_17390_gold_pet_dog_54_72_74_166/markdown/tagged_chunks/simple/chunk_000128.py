from langchain_core.documents import Document

chunk = Document(
    page_content=('- 게 지급합니다.\n'
 '- \uf000 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동\n'
 '- 일하지 않을 때에는 보험금 지급사유가 발생하기 전에 피보험자가 서면(「전자서\n'
 '- 명법」 제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에\n'
 '- 정하는 바에 따라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서\n'
 '- 를 포함)으로 동의하여야 합니다. 또한, 계약자가 보험수익자 변경권을 행사하지\n'
 '- 않고 사망한 경우, 최초 지정된 보험수익자의 권리가 확정됩니다. 그러나 계약자'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000128',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
