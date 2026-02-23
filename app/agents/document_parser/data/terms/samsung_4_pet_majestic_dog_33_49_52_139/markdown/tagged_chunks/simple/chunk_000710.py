from langchain_core.documents import Document

chunk = Document(
    page_content=('- 「보험계약」이라 합니다)을 체결할 때 피보험자의 건강상태가 보험회사(이하「회\n'
 '- 사」라 합니다)가 정한 기준에 적합하지 않은 경우 보험계약자(이하「계약자」라 합니\n'
 '- 다)의 청약과 회사의 승낙으로 보험계약에 부가하여 이루어 집니다.\n'
 '- ② 이 특별약관에 대한 보장개시일(책임개시일)은 보험계약「제1회 보험료 및 회사의 보\n'
 '- 장개시」의 보장개시일(책임개시일)과 동일합니다.\n'
 '- ③ 보험계약이 해지 또는 기타 사유에 의하여 효력이 없게 된 경우에는 이 특별약관도 더\n'
 '- 이상 효력이 없습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000710',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
