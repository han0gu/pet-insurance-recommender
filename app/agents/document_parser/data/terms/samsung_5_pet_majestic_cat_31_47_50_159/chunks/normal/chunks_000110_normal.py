from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보 험가입금액을 감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환급금보다 적어질 수 있습니다. ⑤ 계약자가 제2항에 '
 '따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일 하지 않을 때에는 보험금의 지급사유가 발생하기 전에 피보험자가 '
 '서면(「전자서명 법」제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에 정하는 바에 따라 본인 확인 및 위조ㆍ변조 '
 '방지에 대한 신뢰성을 갖춘 전자문서를 포함)으로 동의하여야 합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
