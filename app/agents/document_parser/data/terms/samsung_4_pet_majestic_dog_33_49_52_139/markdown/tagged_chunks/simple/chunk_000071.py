from langchain_core.documents import Document

chunk = Document(
    page_content=('- 부활(효력회복)된 경우에는 부활(효력회복)계약을 제2항의 최초계약으로 봅니다. 부활\n'
 '- (효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로\n'
 '- 봅니다.\n'
 '# 제19조 (사기에 의한 계약)계약자 또는 피보험자가 대리진단, 약물사용을 수단으로 진단절차를 통과하거나 진단서\n'
 '위·변조 또는 청약일 이전에 암 또는 사람면역결핍바이러스(HIV) 감염의 진단확정을 받\n'
 '은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경'),
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
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
