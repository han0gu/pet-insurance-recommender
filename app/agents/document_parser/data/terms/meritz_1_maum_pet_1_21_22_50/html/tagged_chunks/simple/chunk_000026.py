from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 이 계<br>약이 갱신계약인 경우에는 적용하지 않습니다.<br>④ 회사는 제3항 본문의 면책기간을 최대 30일을 한도로 '
 '적용합니다<br>⑤ 제1항의 「연간」이라 함은 계약일부터 매 1년 단위로 도래하는 계약해당일 전일까지<br>의 기간을 말합니다.<br>➅ '
 '반려동물이 제1항의 질병 또는 상해로 치료를 받던 중에 보험기간이 만료된 경우에도<br>만료일부터 180일 이내의 치료비는 제2항에 따라 '
 '보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
