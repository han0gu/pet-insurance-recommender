from langchain_core.documents import Document

chunk = Document(
    page_content=('마. 유기동물 보호센터 등에서 사육·관리하는 개(犬)\n'
 '② 지급사유 관련 용어\n'
 '1. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 반려견에 입은 상해 를 말하며, 유독 가스 또는 유독 물질을 반려견이 '
 '우연히 일시적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독 증상을 포함합니다. 그러나 음식물 섭취로 인한 증 상, 세균성 음식물 중독과 '
 '상습적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독 증상은 포함되지 않습니다.\n'
 '<용어풀이>\n'
 '[음식물]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 66},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000333',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
