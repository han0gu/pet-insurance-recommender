from langchain_core.documents import Document

chunk = Document(
    page_content=('10. 반려견의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증상을 포함합니다. 다만 보험기간 중 '
 '최초로 발견된 경우에는 보상합니다 )\n'
 '11. 대한민국 이외 지역에서 발생한 사고 및 손해 12. 수의사 자격이 없는 자의 치료행위로 인한 손해(수의사의 소견 및 처방에 의한 '
 '경 우도 동일) 및 그로 인하여 가중된 손해\n'
 '② 회사는 아래의 의료비 및 비용 또는 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
