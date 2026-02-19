from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보 험가입금액을 감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환급금보다 적어질 수 있습니다. ⑤ 계약자가 제2항에 '
 '따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일 하지 않을 때에는 보험금의 지급사유가 발생하기 전에 피보험자가 서면( '
 '「전자서명 버 제2조 제2호에 따르 전자서명이 이느 경으로 서 사버 시해려 제44조의2에 정하느'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
