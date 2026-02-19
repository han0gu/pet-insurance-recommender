from langchain_core.documents import Document

chunk = Document(
    page_content=('8. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으 로 이용함으로써 발생한 손해 9. 동물보호법 '
 '위반 등 동물학대에 기인하는 손해 10. 반려견의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증상을 '
 '포함합니다. 다만 보험기간 중 최초로 발견된 경우에는 보상합니다 .) 11. 대한민국 이외 지역에서 발생한 사고 및 손해 12. 수의사 '
 '자격이 없는 자의 치료행위로 인한 손해(수의사의 소견 및 처방에 의한 경 우도 동일) 및 그로 인하여 가중된 손해'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 115},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000697',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
