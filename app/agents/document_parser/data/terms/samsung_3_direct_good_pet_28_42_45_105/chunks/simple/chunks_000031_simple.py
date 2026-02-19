from langchain_core.documents import Document

chunk = Document(
    page_content=('분쟁조정은 제36조(분쟁의 조정) 제1항에 따라 금융감독원에 신청할 수 있습니다.\n'
 '③ 제2항에 의하여 장해지급률의 판정 및 지급할 보험금의 결정과 관련하여 확정된 장해 지급률에 따른 보험금을 초과한 부분에 대한 분쟁으로 '
 '보험금 지급이 늦어지는 경우 에는 보험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급합니다. ④ 제2항에 의하여 추가적인 조사가 '
 '이루어지는 경우, 회사는 보험수익자의 청구에 따라 회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.\n'
 '<용어풀이>\n'
 '[가지급보험금]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 30},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
