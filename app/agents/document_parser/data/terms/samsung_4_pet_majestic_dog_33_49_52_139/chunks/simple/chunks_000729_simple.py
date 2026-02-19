from langchain_core.documents import Document

chunk = Document(
    page_content=('로 이용함으로써 발생한 손해\n'
 '7. 수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인한 손해(수의사의 소견 및 처방에 의한 경우도 동일) 및 '
 '그로 인하여 가중된 손해 8. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태 9. 원인이 어떠한 경우에도 '
 '반려견에 대한 사료제공 또는 급수 등 기본적인 관리에 대 한 태만 10. 동물보호법 위반 등 동물학대에 기인하는 손해 11. 사망사실을 '
 '명확하게 입증할 수 없는 실종, 행방불명 등\n'
 '제 3조 (보험금의 청구)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000729',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
