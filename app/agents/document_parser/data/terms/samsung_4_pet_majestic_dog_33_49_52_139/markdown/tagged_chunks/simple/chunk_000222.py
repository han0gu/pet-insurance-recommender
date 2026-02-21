from langchain_core.documents import Document

chunk = Document(
    page_content=('약자가 청약서에 자필서명을 하지 않은 때에는 계약자는 계약이 성립한 날부터 3개월\n'
 '이내에 계약을 취소할 수 있습니다.<용어풀이># [자필서명]날인(도장을 찍음) 및 전자서명법 제2조 제2호에 따른 전자서명을 '
 '포함합니다.④ 제3항에도 불구하고 전화를 이용하여 계약을 체결하는 경우 다음의 각 호의 어느 하\n'
 '나를 충족하는 때에는 자필서명을 생략할 수 있으며, 제2항의 규정에 따른 음성녹음\n'
 '내용을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를 전달한 것'),
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
 'indexing': {'chunk_id': 'chunk_000222',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
