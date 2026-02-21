from langchain_core.documents import Document

chunk = Document(
    page_content=('요구받았을 때에는 정당한 사유 없이 이를 거부하여서는 아니된다.\n'
 '④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 서식, 기재사항, 그\n'
 '밖에 필요한 사항은 농림축산식품부령으로 정한다.\n'
 '⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한 축산농장에 상시고용된 수의사와「동물원\n'
 '및 수족관의 관리에 관한 법률」 제8조에 따라 허가받은 동물원 또는 수족관에 상시고용된 수\n'
 '의사는 해당 농장, 동물원 또는 수족관의 동물에게 투여할 목적으로 처방대상 동물용 의약품에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000612',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
