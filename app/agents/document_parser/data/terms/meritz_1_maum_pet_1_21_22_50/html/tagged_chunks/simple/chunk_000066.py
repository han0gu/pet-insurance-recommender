from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 정당한 사유 없이 이에 동의하지 않을 경우 사실 확인이 끝날 때까지 회<br>사는 보험금 지급지연에 따른 이자를 지급하지 '
 '않습니다.<br>⑥ 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고 설명합<br>니다.<br>⑦ 보험수익자와 '
 '회사가 제4조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못<br>할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 '
 '제3자의 의견에 따를 수 있습니<br>다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
