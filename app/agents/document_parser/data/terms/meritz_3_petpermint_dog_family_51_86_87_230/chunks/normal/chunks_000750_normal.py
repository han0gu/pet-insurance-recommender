from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 3개의 척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는 고정한 상태 나) 머리뼈(두개골)와 '
 '제1경추 또는 제1경추와 제2경 추를 유합 또는 고정한 상태 다) 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경 추) 사이에 CT '
 '검사 상, 두개 대후두공의 기저점 (basion)과 축추 치돌기 상단사이의 거리(BDI : Basion-Dental Interval)에 '
 '뚜렷한 이상전위가 있는 상태 라) 상위목뼈(상위경추: 제1, 2경추) CT 검사상, 환 추 전방 궁(arch)의 후방과'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 212},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000750',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
