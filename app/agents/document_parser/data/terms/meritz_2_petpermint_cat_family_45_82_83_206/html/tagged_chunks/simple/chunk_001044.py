from langchain_core.documents import Document

chunk = Document(
    page_content=("이루어지<br>지 않는 ‘불유합’ 상태를 말하며, 골유합이<br>지연되는 지연유합은 제외한다.</p><br><p id='68' "
 'data-category=\'list\' style=\'font-size:20px\'>13) "가관절이 남아 약간의 장해를 남긴 때"라 '
 '함은 경<br>골과 종아리뼈중 어느 한 뼈에 가관절이 남은 경우<br>를 말한다.<br>14) "뼈에 기형을 남긴 때"라 함은 대퇴골 '
 '또는 경골에<br>기형이 남아 정상에 비해 부정유합된 각 변형이<br>15° 이상인 경우를 말한다.<br>15) 다리 길이의 단축 또는 '
 '과신장은'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'joint', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001044',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
