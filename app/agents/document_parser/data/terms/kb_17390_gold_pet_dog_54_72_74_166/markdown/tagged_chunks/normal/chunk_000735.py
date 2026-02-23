from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지급사유) 제3항에 따라 반려동물 위탁비용을 계속 지급합니다.\n'
 '- \uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회\n'
 '- 사는 반려동물 위탁비용의 전부 또는 일부를 지급하지 않습니다.\n'
 '- \uf000 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 질병의 치료를\n'
 '- 목적으로 2회이상 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일수를\n'
 '- 더합니다.\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000735',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
