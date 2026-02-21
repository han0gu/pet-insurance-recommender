from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 제36조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다. 다만, 보\n'
 '- 험가입금액을 감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환급금보다\n'
 '- 적어질 수 있습니다.\n'
 '- ⑤ 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일\n'
 '- 하지 않을 때에는 보험금의 지급사유가 발생하기 전에 피보험자가 서면(「전자서명\n'
 '- 법」제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에 정하는'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
