from langchain_core.documents import Document

chunk = Document(
    page_content=('M50 | 경추간판장애\n'
 'M51 | 기타 추간판장애\n'
 'M54 | 등통증\n'
 '55 | 골반염 | N73 | 기타 여성골반염증질환\n'
 'N74 | 달리 분류된 질환에서의 여성골반염증장애\n'
 '56 | 자궁내막증 | N80 | 자궁내막증\n'
 '57 | 자궁근종 | D25 | 자궁의 평활근종\n'
 '58 | 연골증 | M91 | 고관절 및 골반의 연소성 골연골증\n'
 'M92 | 기타 연소성 골연골증\n'
 'M93 | 기타 골연골병증\n'
 'M94 | 연골의 기타 장애\n'
 '주 ) 제10차 개정 이후 한국표준질병·사인분류에서 상기 분류표에 변경사항이 발생하는 경우에는 변경된 분류표에 따릅니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['joint',
                             'dental',
                             'skin',
                             'urinary',
                             'eye',
                             'digestive']},
 'indexing': {'chunk_id': 'chunk_000870',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
